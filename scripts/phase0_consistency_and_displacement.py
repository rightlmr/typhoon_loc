"""Phase 0 diagnostics: vo_850 consistency and true-vs-field displacement."""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tclocator.common import DomainConfig, haversine_km, iter_files, load_config
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
from tclocator.io_era5 import read_era5_channels
from tclocator.labels import find_field_min_center, read_ibtracs, records_at_time
import pandas as pd


LEAD_BINS = [(0, 24), (24, 48), (48, 96), (96, 120)]


def _output_dir(config: dict[str, Any]) -> Path:
    """Return and create the Phase 0 output directory."""

    out = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs")) / "phase0"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _vo_consistency(config: dict[str, Any]) -> dict[str, Any]:
    """Run the ERA5 vo_850口径 consistency check."""

    era5_cfg = config.get("era5", {})
    era5_files = iter_files(config.get("paths", {}).get("era5_dir", ""), [".nc"])
    if not era5_files:
        return {"status": "no_era5_data"}
    if bool(era5_cfg.get("vo850_from_uv", True)):
        return {"status": "derived_from_uv", "message": "ERA5 与 AIFS 均由 calc_vo850 从 u850/v850 派生"}

    domain = DomainConfig.from_mapping(config.get("domain"))
    direct_cfg = dict(era5_cfg)
    direct_cfg["vo850_from_uv"] = False
    derived_cfg = dict(era5_cfg)
    derived_cfg["vo850_from_uv"] = True
    direct, _ = read_era5_channels(era5_files[0], channels=["vo_850"], domain=domain, era5_config=direct_cfg)
    try:
        derived, _ = read_era5_channels(era5_files[0], channels=["vo_850"], domain=domain, era5_config=derived_cfg)
    except Exception as exc:
        return {
            "status": "precomputed_vo850_without_uv",
            "message": "ERA5 只有预计算 vo_850，缺少 u850/v850，无法自动验证 D5 口径一致性",
            "error": f"{type(exc).__name__}: {exc}",
            "pass": False,
        }
    a = direct[0].ravel()
    b = derived[0].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    peak_ratio = float(np.nanmax(np.abs(a)) / max(float(np.nanmax(np.abs(b))), 1e-12))
    rmse = float(np.sqrt(np.nanmean((a - b) ** 2)))
    return {
        "status": "compared_precomputed_vs_derived",
        "corr": corr,
        "peak_ratio": peak_ratio,
        "rmse": rmse,
        "pass": 0.8 <= peak_ratio <= 1.2,
    }


def _synthetic_displacements() -> pd.DataFrame:
    """Create deterministic smoke-test displacement rows."""

    rows: list[dict[str, Any]] = []
    base_time = pd.Timestamp("2000-08-01T00:00:00Z")
    for sid in ["SYN001", "SYN002"]:
        for lead in [0, 12, 24, 48, 72, 96, 120]:
            displacement = 35.0 + 0.8 * lead + (0.0 if sid == "SYN001" else 8.0)
            rows.append(
                {
                    "SID": sid,
                    "valid_time": base_time + pd.Timedelta(hours=lead),
                    "lead_hour": lead,
                    "lat_true": 15.0,
                    "lon_true": 130.0,
                    "lat_field": 15.0,
                    "lon_field": 130.0 + displacement / 111.0,
                    "displacement_km": displacement,
                }
            )
    return pd.DataFrame(rows)


def _real_displacements(config: dict[str, Any]) -> pd.DataFrame:
    """Collect true-vs-field msl-min displacements from AIFS files."""

    paths = config.get("paths", {})
    aifs_files = iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
    ibtracs_path = Path(paths.get("ibtracs_csv", ""))
    if not aifs_files or not ibtracs_path.exists():
        return pd.DataFrame()

    domain = DomainConfig.from_mapping(config.get("domain"))
    labels_cfg = config.get("labels", {})
    radius = float(labels_cfg.get("phase0_search_radius_km", 500.0))
    records = read_ibtracs(ibtracs_path, config.get("ibtracs", {}).get("col_map", {}))
    rows: list[dict[str, Any]] = []
    for path in aifs_files:
        meta = parse_aifs_filename(path)
        at_time = records_at_time(records, pd.Timestamp(meta.valid_time))
        if at_time.empty:
            continue
        field, _ = read_aifs_channels(path, channels=["msl"], domain=domain, aifs_config=config.get("aifs", {}))
        msl = field[0]
        for _, record in at_time.iterrows():
            lat_field, lon_field = find_field_min_center(msl, float(record["LAT"]), float(record["LON"]), domain, radius)
            displacement = float(haversine_km(record["LAT"], record["LON"], lat_field, lon_field))
            rows.append(
                {
                    "SID": record["SID"],
                    "valid_time": meta.valid_time.isoformat(),
                    "lead_hour": meta.forecast_hour,
                    "lat_true": float(record["LAT"]),
                    "lon_true": float(record["LON"]),
                    "lat_field": lat_field,
                    "lon_field": lon_field,
                    "displacement_km": displacement,
                }
            )
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize displacement by lead bins."""

    rows: list[dict[str, Any]] = []
    for lo, hi in LEAD_BINS:
        part = df.loc[(df["lead_hour"] >= lo) & (df["lead_hour"] < hi)]
        if part.empty:
            continue
        rows.append(
            {
                "lead_bin": f"{lo:03d}-{hi:03d}",
                "n": int(len(part)),
                "mean_km": float(part["displacement_km"].mean()),
                "median_km": float(part["displacement_km"].median()),
                "p75_km": float(part["displacement_km"].quantile(0.75)),
                "p90_km": float(part["displacement_km"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, path: Path) -> None:
    """Write a simple displacement-vs-lead scatter PNG without GUI backends."""

    width, height = 900, 560
    margin_left, margin_bottom, margin_top, margin_right = 80, 70, 40, 30
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    def draw_rect(cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        y0 = max(0, cy - radius)
        y1 = min(height, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(width, cx + radius + 1)
        image[y0:y1, x0:x1] = color

    def draw_line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, n + 1).astype(int)
        ys = np.linspace(y0, y1, n + 1).astype(int)
        image[np.clip(ys, 0, height - 1), np.clip(xs, 0, width - 1)] = color

    plot_x0 = margin_left
    plot_x1 = width - margin_right
    plot_y0 = margin_top
    plot_y1 = height - margin_bottom
    draw_line(plot_x0, plot_y1, plot_x1, plot_y1, (0, 0, 0))
    draw_line(plot_x0, plot_y0, plot_x0, plot_y1, (0, 0, 0))

    x_values = df["lead_hour"].astype(float).to_numpy()
    y_values = df["displacement_km"].astype(float).to_numpy()
    x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    y_min, y_max = 0.0, float(max(np.nanmax(y_values), 1.0))
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    for frac in np.linspace(0.0, 1.0, 6):
        x = int(plot_x0 + frac * (plot_x1 - plot_x0))
        y = int(plot_y1 - frac * (plot_y1 - plot_y0))
        draw_line(x, plot_y0, x, plot_y1, (230, 230, 230))
        draw_line(plot_x0, y, plot_x1, y, (230, 230, 230))

    for lead, displacement in zip(x_values, y_values, strict=True):
        x = int(plot_x0 + (lead - x_min) / (x_max - x_min) * (plot_x1 - plot_x0))
        y = int(plot_y1 - (displacement - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))
        draw_rect(x, y, 3, (31, 119, 180))

    _write_png(path, image)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image as PNG using the standard library."""

    height, width, channels = image.shape
    if channels != 3 or image.dtype != np.uint8:
        raise ValueError("PNG writer expects RGB uint8 image")

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    raw_rows = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw_rows, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _suggest(df: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    """Suggest labels.mode, lead_max, and search radius from Phase 0 results."""

    if df.empty:
        return {"labels.mode": None, "finetune.lead_max": None, "labels.search_radius_km": 300}
    all_small = not summary.empty and bool((summary["median_km"] < 75.0).all())
    labels_mode = "ibtracs" if all_small else "in_field"
    acceptable = summary.loc[summary["median_km"] <= 75.0]
    if acceptable.empty:
        lead_max = int(min(24, df["lead_hour"].max()))
    else:
        lead_max = int(acceptable["lead_bin"].iloc[-1].split("-")[1])
    p90 = float(df["displacement_km"].quantile(0.90))
    radius = int(max(300, np.ceil(p90 / 50.0) * 50))
    return {"labels.mode": labels_mode, "finetune.lead_max": lead_max, "labels.search_radius_km": radius}


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--smoke-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = _output_dir(config)
    vo_result = {"status": "synthetic_smoke"} if args.smoke_synthetic else _vo_consistency(config)
    df = _synthetic_displacements() if args.smoke_synthetic else _real_displacements(config)
    if df.empty:
        print("Phase 0 displacement skipped: no AIFS/IBTrACS data found.")
    else:
        raw_csv = out_dir / "displacement_vs_lead.csv"
        summary_csv = out_dir / "displacement_summary_by_lead.csv"
        png = out_dir / "displacement_vs_lead.png"
        summary = _summarize(df)
        df.to_csv(raw_csv, index=False)
        summary.to_csv(summary_csv, index=False)
        _plot(df, png)
        print(f"Wrote {raw_csv}")
        print(f"Wrote {summary_csv}")
        print(f"Wrote {png}")

    summary = _summarize(df) if not df.empty else pd.DataFrame()
    suggestions = _suggest(df, summary)
    print(f"vo_850 consistency: {vo_result}")
    print(f"建议 labels.mode = {suggestions['labels.mode']}")
    print(f"建议 finetune.lead_max = {suggestions['finetune.lead_max']}")
    print(f"建议 labels.search_radius_km = {suggestions['labels.search_radius_km']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
