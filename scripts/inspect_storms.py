"""Inspect AIFS storm signal near IBTrACS truth and model peaks."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
import numpy as np
import pandas as pd
import torch

from tclocator.common import (
    DomainConfig,
    build_lat_lon,
    grid_to_latlon,
    haversine_km,
    in_domain,
    iter_files,
    latlon_to_grid,
    load_config,
    resolve_device,
    set_seed,
)
from tclocator.decode import decode_heatmap
from tclocator.labels import read_ibtracs, records_at_time
from tclocator.model import build_model_from_config
from tclocator.normalization import apply_norm, load_norm_stats
from tclocator.split import select_aifs_files


OUTPUT_COLUMNS = [
    "sid",
    "valid_time",
    "lead_hour",
    "true_lat",
    "true_lon",
    "pred_lat",
    "pred_lon",
    "pred_conf",
    "dist_pred_truth_km",
    "msl_at_truth_pa",
    "msl_min_pa",
    "msl_min_dist_km",
    "vo_at_truth",
    "vo_max",
    "vo_max_dist_km",
]


def _load_model(config: dict[str, Any], device: str) -> torch.nn.Module:
    """Load the configured fine-tuned checkpoint."""

    model = build_model_from_config(config).to(device)
    ckpt_path = Path(config.get("paths", {}).get("finetune_ckpt", ""))
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(payload.get("model_state", payload), strict=True)
        print(f"Loaded {ckpt_path}")
    else:
        print(f"Checkpoint not found: {ckpt_path}; running with random weights.")
    model.eval()
    return model


def _aifs_val_files(config: dict[str, Any]) -> list[Path]:
    """Return AIFS validation files, preferring the recorded split JSON."""

    paths = config.get("paths", {})
    files = iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
    split_path = Path(paths.get("output_dir", ROOT / "outputs")) / "split_aifs.json"
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        val_groups = {str(group) for group in payload.get("val_groups", [])}
        if val_groups:
            return [path for path in files if parse_aifs_filename(path).init_time.isoformat() in val_groups]
    return select_aifs_files(config, files, "val")


def _top_prediction(
    model: torch.nn.Module,
    field: np.ndarray,
    config: dict[str, Any],
    domain: DomainConfig,
    device: str,
) -> tuple[float | None, float | None, float | None]:
    """Return the highest-confidence decoded model peak."""

    tensor = torch.as_tensor(field, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
    decoded = decode_heatmap(
        outputs["heatmap"][0, 0],
        outputs["offset"][0],
        domain,
        conf_thresh=0.0,
        lat_filter=tuple(config.get("decode", {}).get("lat_filter", [0.0, 40.0])),
        topk=1,
    )
    if decoded.empty:
        return None, None, None
    row = decoded.iloc[0]
    return float(row["LAT"]), float(row["LON"]), float(row["CONF"])


def _truth_grid_value(arr: np.ndarray, true_lat: float, true_lon: float, domain: DomainConfig) -> float:
    """Return the nearest-grid value at the truth position."""

    y_f, x_f = latlon_to_grid(true_lat, true_lon, domain, clip=True)
    y = int(np.clip(round(float(y_f)), 0, domain.height - 1))
    x = int(np.clip(round(float(x_f)), 0, domain.width - 1))
    return float(arr[y, x])


def _local_extreme(
    arr: np.ndarray,
    true_lat: float,
    true_lon: float,
    domain: DomainConfig,
    *,
    radius_km: float,
    mode: str,
) -> tuple[float, float, float, float]:
    """Return local extreme value, location, and distance to truth."""

    lat1d, lon1d = build_lat_lon(domain)
    dist = haversine_km(lat1d[:, None], lon1d[None, :], true_lat, true_lon)
    mask = dist <= radius_km
    if not np.any(mask):
        raise ValueError("No grid points inside local extreme radius")
    filled = np.where(mask, arr, np.inf if mode == "min" else -np.inf)
    if mode == "min":
        y, x = np.unravel_index(int(np.argmin(filled)), filled.shape)
    elif mode == "max":
        y, x = np.unravel_index(int(np.argmax(filled)), filled.shape)
    else:
        raise ValueError("mode must be 'min' or 'max'")
    lat, lon = grid_to_latlon(float(y), float(x), domain)
    return float(filled[y, x]), float(lat), float(lon), float(dist[y, x])


def _sanitize_name(value: str) -> str:
    """Return a filesystem-safe name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _maybe_plot_case(
    *,
    out_dir: Path,
    case_index: int,
    row: dict[str, Any],
    msl: np.ndarray,
    vo: np.ndarray,
    domain: DomainConfig,
) -> None:
    """Write an optional diagnostic PNG for one case when matplotlib exists."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if case_index == 0:
            print("matplotlib is not installed; skipping diagnostic PNG output.")
        return

    lat1d, lon1d = build_lat_lon(domain)
    true_lat = float(row["true_lat"])
    true_lon = float(row["true_lon"])
    lat_mask = (lat1d >= true_lat - 8.0) & (lat1d <= true_lat + 8.0)
    lon_mask = (lon1d >= true_lon - 8.0) & (lon1d <= true_lon + 8.0)
    if not np.any(lat_mask) or not np.any(lon_mask):
        return
    ys = np.where(lat_mask)[0]
    xs = np.where(lon_mask)[0]
    msl_crop = msl[np.ix_(ys, xs)]
    vo_crop = vo[np.ix_(ys, xs)]
    extent = [float(lon1d[xs[0]]), float(lon1d[xs[-1]]), float(lat1d[ys[-1]]), float(lat1d[ys[0]])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, data, title in ((axes[0], msl_crop, "msl"), (axes[1], vo_crop, "vo_850")):
        image = ax.imshow(data, extent=extent, origin="upper", aspect="auto")
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.scatter([true_lon], [true_lat], marker="*", s=80, c="yellow", edgecolors="black", label="truth")
        if not math.isnan(float(row.get("pred_lat") or float("nan"))):
            ax.scatter([float(row["pred_lon"])], [float(row["pred_lat"])], marker="x", s=70, c="red", label="pred")
        if title == "msl":
            ax.scatter([float(row["msl_min_lon"])], [float(row["msl_min_lat"])], marker="o", s=50, facecolors="none", edgecolors="white", label="msl min")
        else:
            ax.scatter([float(row["vo_max_lon"])], [float(row["vo_max_lat"])], marker="o", s=50, facecolors="none", edgecolors="white", label="vo max")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

    name = _sanitize_name(f"{case_index:02d}_{row['sid']}_{row['valid_time']}_f{int(row['lead_hour']):03d}.png")
    fig.savefig(out_dir / name, dpi=150)
    plt.close(fig)


def _inspect_cases(config: dict[str, Any], max_cases: int, *, emit_plots: bool) -> pd.DataFrame:
    """Build the storm-signal diagnostic table."""

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    domain = DomainConfig.from_mapping(config.get("domain"))
    model = _load_model(config, device)
    norm_path = Path(config.get("paths", {}).get("norm_stats_aifs", ""))
    norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None
    records = read_ibtracs(config.get("paths", {}).get("ibtracs_csv", ""), config.get("ibtracs", {}).get("col_map", {}))
    lead_max = int(config.get("finetune", {}).get("lead_max", 24))
    out_dir = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs")) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for path in _aifs_val_files(config):
        meta = parse_aifs_filename(path)
        if meta.forecast_hour > lead_max:
            continue
        at_time = records_at_time(records, pd.Timestamp(meta.valid_time))
        if at_time.empty:
            continue
        signal_field, _ = read_aifs_channels(path, channels=["msl", "vo_850"], domain=domain, aifs_config=config.get("aifs", {}))
        msl = signal_field[0]
        vo = signal_field[1]
        model_field, _ = read_aifs_channels(path, channels=config["channels"], domain=domain, aifs_config=config.get("aifs", {}))
        if norm_stats is not None:
            model_field = apply_norm(model_field, norm_stats)
        pred_lat, pred_lon, pred_conf = _top_prediction(model, model_field, config, domain, device)

        for _, record in at_time.iterrows():
            true_lat = float(record["LAT"])
            true_lon = float(record["LON"])
            if not in_domain(true_lat, true_lon, domain):
                continue
            msl_min, msl_min_lat, msl_min_lon, msl_min_dist = _local_extreme(
                msl,
                true_lat,
                true_lon,
                domain,
                radius_km=200.0,
                mode="min",
            )
            vo_max, vo_max_lat, vo_max_lon, vo_max_dist = _local_extreme(
                vo,
                true_lat,
                true_lon,
                domain,
                radius_km=200.0,
                mode="max",
            )
            dist_pred = (
                float(haversine_km(pred_lat, pred_lon, true_lat, true_lon))
                if pred_lat is not None and pred_lon is not None
                else None
            )
            row = {
                "sid": str(record["SID"]),
                "valid_time": meta.valid_time.isoformat(),
                "lead_hour": int(meta.forecast_hour),
                "true_lat": true_lat,
                "true_lon": true_lon,
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "pred_conf": pred_conf,
                "dist_pred_truth_km": dist_pred,
                "msl_at_truth_pa": _truth_grid_value(msl, true_lat, true_lon, domain),
                "msl_min_pa": msl_min,
                "msl_min_dist_km": msl_min_dist,
                "vo_at_truth": _truth_grid_value(vo, true_lat, true_lon, domain),
                "vo_max": vo_max,
                "vo_max_dist_km": vo_max_dist,
                "msl_min_lat": msl_min_lat,
                "msl_min_lon": msl_min_lon,
                "vo_max_lat": vo_max_lat,
                "vo_max_lon": vo_max_lon,
            }
            if emit_plots:
                _maybe_plot_case(out_dir=out_dir, case_index=len(rows), row=row, msl=msl, vo=vo, domain=domain)
            rows.append(row)
            if len(rows) >= max_cases:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--plots", action="store_true", help="Write optional PNG diagnostics with matplotlib.")
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.plots:
        print("PNG diagnostics disabled; pass --plots to enable optional matplotlib output.")
    df = _inspect_cases(config, max(0, int(args.max_cases)), emit_plots=bool(args.plots))
    out_dir = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs")) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "storm_signal.csv"
    public = df[[col for col in OUTPUT_COLUMNS if col in df.columns]] if not df.empty else pd.DataFrame(columns=OUTPUT_COLUMNS)
    public.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"cases={len(public)}")
    print(f"median msl_min_dist_km = {public['msl_min_dist_km'].median() if not public.empty else float('nan'):.2f}")
    print(f"median vo_max_dist_km  = {public['vo_max_dist_km'].median() if not public.empty else float('nan'):.2f}")
    print(f"median dist_pred_truth_km = {public['dist_pred_truth_km'].median() if not public.empty else float('nan'):.2f}")
    return 0 if len(public) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
