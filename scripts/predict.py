"""Run TC locator inference and write decoded detections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
import pandas as pd
import torch

from tclocator.common import DomainConfig, iter_files, load_config, resolve_device, set_seed
from tclocator.dataset import SyntheticTCDataset
from tclocator.decode import decode_heatmap
from tclocator.io_era5 import read_era5_channels
from tclocator.model import build_model_from_config
from tclocator.normalization import apply_norm, load_norm_stats
from tclocator.split import select_aifs_files


def _load_model(config: dict[str, Any], device: str, checkpoint: Path | None) -> torch.nn.Module:
    """Load a model checkpoint if provided."""

    model = build_model_from_config(config).to(device)
    if checkpoint is not None and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model_state", payload), strict=True)
        print(f"Loaded {checkpoint}")
    elif checkpoint is not None:
        print(f"Checkpoint not found: {checkpoint}; running with random weights.")
    model.eval()
    return model


def _predict_array(
    model: torch.nn.Module,
    field: object,
    config: dict[str, Any],
    device: str,
    *,
    iso_time: str | None,
    lead_hour: int | None,
    conf_thresh: float | None = None,
) -> pd.DataFrame:
    """Predict one already-normalized field array."""

    domain = DomainConfig.from_mapping(config.get("domain"))
    tensor = torch.as_tensor(field, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    return decode_heatmap(
        out["heatmap"][0, 0],
        out["offset"][0],
        domain,
        iso_time=iso_time,
        lead_hour=lead_hour,
        conf_thresh=float(conf_thresh if conf_thresh is not None else config.get("decode", {}).get("conf_thresh", 0.3)),
        lat_filter=tuple(config.get("decode", {}).get("lat_filter", [0.0, 40.0])),
    )


def _with_split_suffix(path: Path, split: str) -> Path:
    """Return an output path with a split suffix when requested."""

    if split == "all":
        return path
    return path.with_name(f"{path.stem}_{split}{path.suffix}")


def _lead_max_from_config(config: dict[str, Any]) -> int | None:
    """Return configured AIFS forecast lead limit, if present."""

    raw = config.get("finetune", {}).get("lead_max")
    return int(raw) if raw is not None else None


def _filter_by_lead_max(files: list[Path], lead_max: int | None) -> list[Path]:
    """Filter AIFS files to the configured maximum forecast lead."""

    if lead_max is None:
        return files
    kept: list[Path] = []
    for path in files:
        if parse_aifs_filename(path).forecast_hour <= lead_max:
            kept.append(path)
    return kept


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--domain", choices=["aifs", "era5"], default="aifs")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--conf-thresh", type=float, default=None)
    parser.add_argument("--smoke-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(config.get("paths", {}).get("finetune_ckpt", ""))
    model = _load_model(config, device, ckpt)
    rows: list[pd.DataFrame] = []

    if args.smoke_synthetic:
        dataset = SyntheticTCDataset(length=2, channels=config["channels"], seed=int(config.get("seed", 42)))
        for idx in range(len(dataset)):
            rows.append(
                _predict_array(
                    model,
                    dataset[idx]["input"].numpy(),
                    config,
                    device,
                    iso_time=f"synthetic_{idx}",
                    lead_hour=0,
                    conf_thresh=args.conf_thresh,
                )
            )
    elif args.domain == "aifs":
        norm_path = Path(config.get("paths", {}).get("norm_stats_aifs", ""))
        norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None
        files = iter_files(config.get("paths", {}).get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
        files = select_aifs_files(config, files, args.split)
        files = _filter_by_lead_max(files, _lead_max_from_config(config))
        domain_cfg = DomainConfig.from_mapping(config.get("domain"))
        for index, path in enumerate(files, start=1):
            if index % 200 == 0:
                print(f"predict aifs processed={index}/{len(files)} split={args.split}", flush=True)
            field, meta = read_aifs_channels(path, channels=config["channels"], domain=domain_cfg, aifs_config=config.get("aifs", {}))
            if norm_stats is not None:
                field = apply_norm(field, norm_stats)
            parsed = parse_aifs_filename(path)
            rows.append(
                _predict_array(
                    model,
                    field,
                    config,
                    device,
                    iso_time=parsed.valid_time.isoformat(),
                    lead_hour=parsed.forecast_hour,
                    conf_thresh=args.conf_thresh,
                )
            )
    else:
        norm_path = Path(config.get("paths", {}).get("norm_stats_era5", ""))
        norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None
        files = iter_files(config.get("paths", {}).get("era5_dir", ""), [".nc"])
        domain_cfg = DomainConfig.from_mapping(config.get("domain"))
        for path in files:
            field, _ = read_era5_channels(path, channels=config["channels"], domain=domain_cfg, era5_config=config.get("era5", {}))
            if norm_stats is not None:
                field = apply_norm(field, norm_stats)
            rows.append(_predict_array(model, field, config, device, iso_time=path.stem, lead_hour=None, conf_thresh=args.conf_thresh))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ISO_TIME", "LAT", "LON", "CONF"])
    out_path = Path(args.output) if args.output else Path(config.get("paths", {}).get("predictions_csv", ROOT / "outputs" / "predictions.csv"))
    if not args.output:
        out_path = _with_split_suffix(out_path, args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
