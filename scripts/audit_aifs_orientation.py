"""Audit spatial orientation of serialized AIFS .pt fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
from tclocator.io_aifs import DEFAULT_TENSOR_CHANNEL_ORDER, parse_aifs_filename
import numpy as np
import torch

from tclocator.common import iter_files, load_config


def _load_tensor(path: Path) -> np.ndarray:
    """Load an AIFS .pt payload as a NumPy array."""

    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        for candidate in ("tensor", "data", "field", "fields"):
            if candidate in tensor:
                tensor = tensor[candidate]
                break
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    if arr.ndim != 3:
        raise ValueError(f"AIFS .pt tensor must be [C,H,W], got {arr.shape}")
    return np.asarray(arr)


def _find_reference_file(config: dict[str, Any], ref_valid: str) -> Path:
    """Find the lead-0 AIFS file for a reference valid time."""

    target = np.datetime64(ref_valid)
    files = iter_files(config.get("paths", {}).get("aifs_dir", ""), [".pt"])
    for path in files:
        meta = parse_aifs_filename(path)
        if meta.forecast_hour == 0 and np.datetime64(meta.valid_time.replace(tzinfo=None)) == target:
            return path
    raise FileNotFoundError(f"No lead-0 AIFS .pt file found for valid_time={ref_valid}")


def _sample_3x3(arr: np.ndarray, row: int, col: int) -> float:
    """Return a wrapped 3x3 mean around one grid point."""

    rows = [(row + delta) % arr.shape[0] for delta in (-1, 0, 1)]
    cols = [(col + delta) % arr.shape[1] for delta in (-1, 0, 1)]
    return float(np.mean(arr[np.ix_(rows, cols)]))


def _row_for_lat(lat: float, lat_order: str, height: int) -> int:
    """Return row index under one latitude-order hypothesis."""

    if lat_order == "north_first":
        row = round((90.0 - lat) / 0.25)
    elif lat_order == "south_first":
        row = round((lat + 90.0) / 0.25)
    else:
        raise ValueError(f"Unsupported lat_order: {lat_order}")
    return int(row) % height


def _col_for_lon(lon: float, lon_mode: str, width: int) -> int:
    """Return column index under one longitude-origin hypothesis."""

    if lon_mode == "from_0":
        col = round((lon % 360.0) / 0.25)
    elif lon_mode in {"roll_180", "from_180"}:
        col = round(((lon + 180.0) % 360.0) / 0.25)
    else:
        raise ValueError(f"Unsupported lon_mode: {lon_mode}")
    return int(col) % width


def _audit_array(msl: np.ndarray, ref_lat: float, ref_lon: float, *, label: str) -> list[dict[str, Any]]:
    """Sample MSL under orientation hypotheses."""

    rows: list[dict[str, Any]] = []
    for lat_order in ("north_first", "south_first"):
        for lon_mode in ("from_0", "roll_180", "from_180"):
            row = _row_for_lat(ref_lat, lat_order, msl.shape[0])
            col = _col_for_lon(ref_lon, lon_mode, msl.shape[1])
            value = _sample_3x3(msl, row, col)
            rows.append(
                {
                    "array": label,
                    "lat_order": lat_order,
                    "lon_mode": lon_mode,
                    "row": row,
                    "col": col,
                    "sampled_msl_hPa": value / 100.0,
                }
            )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print audit rows in a compact table."""

    print("array\tlat_order\tlon_mode\trow\tcol\tsampled_msl_hPa")
    for row in rows:
        print(
            f"{row['array']}\t{row['lat_order']}\t{row['lon_mode']}\t"
            f"{row['row']}\t{row['col']}\t{row['sampled_msl_hPa']:.2f}"
        )


def _global_min_report(msl: np.ndarray) -> None:
    """Print global MSL minimum under the current north_first/from_0 convention."""

    y, x = np.unravel_index(int(np.nanargmin(msl)), msl.shape)
    lat = 90.0 - float(y) * 0.25
    lon = float(x) * 0.25
    print(
        "global_argmin_north_first_from_0: "
        f"row={y} col={x} lat={lat:.2f} lon={lon:.2f} msl_hPa={float(msl[y, x]) / 100.0:.2f}"
    )


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--ref-file", default=None)
    parser.add_argument("--ref-lat", type=float, default=21.0)
    parser.add_argument("--ref-lon", type=float, default=106.0)
    parser.add_argument("--ref-valid", default="2024-09-07T12:00:00")
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(args.ref_file) if args.ref_file else _find_reference_file(config, args.ref_valid)
    tensor = _load_tensor(path)
    order = [name.replace("mslp", "msl") for name in config.get("aifs", {}).get("tensor_channel_order", DEFAULT_TENSOR_CHANNEL_ORDER)]
    msl_idx = order.index("msl")
    msl = np.asarray(tensor[msl_idx], dtype=np.float32)
    print(f"file={path}")
    print(f"tensor_shape={tuple(tensor.shape)} msl_index={msl_idx}")
    print(f"msl_raw_min_hPa={float(np.nanmin(msl)) / 100.0:.2f} max_hPa={float(np.nanmax(msl)) / 100.0:.2f} mean_hPa={float(np.nanmean(msl)) / 100.0:.2f}")

    rows = _audit_array(msl, float(args.ref_lat), float(args.ref_lon), label="raw")
    if msl.shape == (1440, 721):
        rows.extend(_audit_array(msl.T, float(args.ref_lat), float(args.ref_lon), label="transpose"))
    elif msl.shape == (721, 1440):
        rows.extend(_audit_array(msl.T, float(args.ref_lat), float(args.ref_lon), label="transpose_probe"))
    _print_table(rows)

    candidates = [row for row in rows if row["sampled_msl_hPa"] < 990.0]
    if candidates:
        winner = min(candidates, key=lambda row: row["sampled_msl_hPa"])
        print(
            "WINNER: "
            f"array={winner['array']} lat_order={winner['lat_order']} lon_mode={winner['lon_mode']} "
            f"sampled_msl_hPa={winner['sampled_msl_hPa']:.2f}"
        )
    else:
        best = min(rows, key=lambda row: row["sampled_msl_hPa"])
        print(
            "NO_DEEP_LOW_WINNER: "
            f"best array={best['array']} lat_order={best['lat_order']} lon_mode={best['lon_mode']} "
            f"sampled_msl_hPa={best['sampled_msl_hPa']:.2f}"
        )
    _global_min_report(msl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
