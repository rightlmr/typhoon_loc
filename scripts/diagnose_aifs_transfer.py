"""Diagnose AIFS transfer behavior without overwriting training outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
import numpy as np
import pandas as pd
import torch

from tclocator.common import DomainConfig, grid_to_latlon, haversine_km, iter_files, load_config, resolve_device, set_seed
from tclocator.dataset import build_aifs_samples
from tclocator.decode import decode_heatmap
from tclocator.labels import find_field_min_center, load_label_npz, read_ibtracs, records_at_time
from tclocator.losses import LossConfig, TCLocatorLoss
from tclocator.metrics import match_predictions, precision_recall_curve, summarize_by_lead
from tclocator.model import build_model_from_config
from tclocator.normalization import apply_norm, load_norm_stats


def build_aifs_references(config: dict[str, Any]) -> pd.DataFrame:
    """Build AIFS references with field-consistent centers."""

    domain = DomainConfig.from_mapping(config.get("domain"))
    records = read_ibtracs(config["paths"]["ibtracs_csv"], config.get("ibtracs", {}).get("col_map", {}))
    rows: list[dict[str, Any]] = []
    files = iter_files(config["paths"]["aifs_dir"], [".pt", ".grib2", ".grb2", ".grib"])
    for index, path in enumerate(files, start=1):
        meta = parse_aifs_filename(path)
        at_time = records_at_time(records, pd.Timestamp(meta.valid_time))
        if at_time.empty:
            continue
        field, _ = read_aifs_channels(path, channels=["msl"], domain=domain, aifs_config=config.get("aifs", {}))
        for _, record in at_time.iterrows():
            lat_field, lon_field = find_field_min_center(
                field[0],
                float(record["LAT"]),
                float(record["LON"]),
                domain,
                float(config.get("labels", {}).get("search_radius_km", 300.0)),
            )
            rows.append(
                {
                    "ISO_TIME": meta.valid_time.isoformat(),
                    "SID": record["SID"],
                    "LEAD_HOUR": meta.forecast_hour,
                    "LAT_TRUE": float(record["LAT"]),
                    "LON_TRUE": float(record["LON"]),
                    "LAT_FIELD": lat_field,
                    "LON_FIELD": lon_field,
                }
            )
        if index % 200 == 0:
            print(f"processed={index} refs={len(rows)}", flush=True)
    return pd.DataFrame(rows)


def recompute_metrics(config: dict[str, Any], out_dir: Path) -> None:
    """Recompute M3/M4 metrics with the current matching implementation."""

    references = build_aifs_references(config)
    references.to_csv(out_dir / "aifs_references.csv", index=False)
    print(f"references={len(references)}", flush=True)
    for tag, pred_path in {
        "m3": ROOT / "outputs" / "m3_pretrain_baseline" / "predictions.csv",
        "m4": ROOT / "outputs" / "predictions.csv",
    }.items():
        predictions = pd.read_csv(pred_path)
        matched = match_predictions(predictions, references)
        summary = summarize_by_lead(matched)
        pr = precision_recall_curve(predictions, references, thresholds=[0.1, 0.2, 0.3, 0.5, 0.7])
        matched.to_csv(out_dir / f"{tag}_matched_metrics_fixed.csv", index=False)
        summary.to_csv(out_dir / f"{tag}_metrics_by_lead_fixed.csv", index=False)
        pr.to_csv(out_dir / f"{tag}_precision_recall_fixed.csv", index=False)
        print(f"\n{tag} metrics", flush=True)
        print(summary.to_string(index=False), flush=True)
        print(pr.to_string(index=False), flush=True)


def load_model(config: dict[str, Any], checkpoint: Path, device: str) -> torch.nn.Module:
    """Load one locator checkpoint."""

    model = build_model_from_config(config).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload.get("model_state", payload), strict=True)
    model.eval()
    return model


def aifs_samples_for_overfit(config: dict[str, Any], n_samples: int) -> list[dict[str, Any]]:
    """Load a few short-lead AIFS samples into memory."""

    domain = DomainConfig.from_mapping(config.get("domain"))
    norm_stats = load_norm_stats(config["paths"]["norm_stats_aifs"])
    files = iter_files(config["paths"]["aifs_dir"], [".pt", ".grib2", ".grb2", ".grib"])
    samples = build_aifs_samples(files, lead_max=int(config["finetune"]["lead_max"]), config=config)[:n_samples]
    items: list[dict[str, Any]] = []
    for sample in samples:
        field, _ = read_aifs_channels(
            sample.path,
            channels=config["channels"],
            domain=domain,
            aifs_config=config.get("aifs", {}),
        )
        label = load_label_npz(sample.label_path)
        items.append(
            {
                "name": sample.path.name,
                "input": torch.from_numpy(apply_norm(field, norm_stats)).float(),
                "heatmap": torch.from_numpy(label["heatmap"][None]).float(),
                "offset": torch.from_numpy(label["offset"]).float(),
                "mask": torch.from_numpy(label["mask"]).float(),
            }
        )
    return items


def evaluate_tiny_set(model: torch.nn.Module, items: list[dict[str, Any]], config: dict[str, Any], device: str) -> dict[str, float]:
    """Evaluate top-1 distance and heatmap response on loaded tiny samples."""

    domain = DomainConfig.from_mapping(config.get("domain"))
    model.eval()
    label_scores: list[float] = []
    top_dists: list[float] = []
    top_hits = 0
    with torch.no_grad():
        for item in items:
            outputs = model(item["input"].unsqueeze(0).to(device))
            heatmap = outputs["heatmap"][0, 0].detach().cpu().numpy()
            offset = outputs["offset"][0].detach().cpu().numpy()
            mask = item["mask"].numpy()
            target_offset = item["offset"].numpy()
            ys, xs = np.where(mask > 0.5)
            refs = [
                grid_to_latlon(float(y) + float(target_offset[0, y, x]), float(x) + float(target_offset[1, y, x]), domain)
                for y, x in zip(ys, xs, strict=True)
            ]
            label_scores.extend(float(heatmap[y, x]) for y, x in zip(ys, xs, strict=True))
            decoded = decode_heatmap(heatmap, offset, domain, conf_thresh=0.01, lat_filter=(0.0, 40.0), topk=1)
            if decoded.empty or not refs:
                continue
            top = decoded.iloc[0]
            dist = min(
                float(haversine_km(float(top["LAT"]), float(top["LON"]), float(lat), float(lon))) for lat, lon in refs
            )
            top_dists.append(dist)
            top_hits += int(dist < 100.0)
    return {
        "label_mean": float(np.mean(label_scores)) if label_scores else float("nan"),
        "label_max": float(np.max(label_scores)) if label_scores else float("nan"),
        "top_median_km": float(np.median(top_dists)) if top_dists else float("inf"),
        "top_hits_lt100": float(top_hits),
        "n_top": float(len(top_dists)),
    }


def tiny_overfit(config: dict[str, Any], out_dir: Path, n_samples: int, steps: int) -> None:
    """Run a non-destructive tiny AIFS overfit diagnostic."""

    set_seed(123)
    device = resolve_device(str(config.get("device", "auto")))
    items = aifs_samples_for_overfit(config, n_samples)
    model = load_model(config, Path(config["paths"]["pretrain_ckpt"]), device)
    model.freeze_encoder()
    loss_fn = TCLocatorLoss(LossConfig.from_config(config))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    rows: list[dict[str, float | int]] = []
    checkpoints = {0, 25, 50, 100, steps}
    for step in range(steps + 1):
        if step in checkpoints:
            metrics = evaluate_tiny_set(model, items, config, device)
            rows.append({"step": step, **metrics})
            print(
                f"tiny step={step} label_mean={metrics['label_mean']:.5f} "
                f"top_median_km={metrics['top_median_km']:.1f} "
                f"top<100={int(metrics['top_hits_lt100'])}/{int(metrics['n_top'])}",
                flush=True,
            )
        if step == steps:
            break
        item = items[step % len(items)]
        batch = {key: value.unsqueeze(0).to(device) for key, value in item.items() if isinstance(value, torch.Tensor)}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(model(batch["input"]), batch)
        losses["loss"].backward()
        optimizer.step()
    pd.DataFrame(rows).to_csv(out_dir / "tiny_overfit.csv", index=False)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "diagnostics"))
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_metrics:
        recompute_metrics(config, out_dir)
    if not args.skip_overfit:
        tiny_overfit(config, out_dir, args.n_samples, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
