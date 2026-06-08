"""IBTrACS-driven field-consistent label generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from tclocator.common import DomainConfig, grid_to_latlon, haversine_km, in_domain, latlon_to_grid


def read_ibtracs(path: str | Path, col_map: Mapping[str, str]) -> pd.DataFrame:
    """Read and normalize an IBTrACS CSV."""

    df = pd.read_csv(path)
    required = [col_map["time"], col_map["sid"], col_map["lat"], col_map["lon"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"IBTrACS CSV is missing columns: {missing}")
    out = pd.DataFrame(
        {
            "ISO_TIME": pd.to_datetime(df[col_map["time"]], utc=True, errors="coerce"),
            "SID": df[col_map["sid"]].astype(str),
            "LAT": pd.to_numeric(df[col_map["lat"]], errors="coerce"),
            "LON": pd.to_numeric(df[col_map["lon"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ISO_TIME", "LAT", "LON"]).reset_index(drop=True)
    out["LON"] = np.mod(out["LON"].astype(float), 360.0)
    return out


def records_at_time(records: pd.DataFrame, valid_time: pd.Timestamp | str) -> pd.DataFrame:
    """Return IBTrACS records exactly matching a valid time."""

    time = pd.Timestamp(valid_time)
    if time.tzinfo is None:
        time = time.tz_localize("UTC")
    else:
        time = time.tz_convert("UTC")
    return records.loc[records["ISO_TIME"] == time].copy()


def _box_smooth(arr: np.ndarray, radius: int) -> np.ndarray:
    """Return a simple edge-padded box mean using NumPy only."""

    if radius <= 0:
        return arr
    kernel = 2 * radius + 1
    padded = np.pad(arr, radius, mode="edge")
    out = np.zeros_like(arr, dtype=np.float64)
    for dy in range(kernel):
        for dx in range(kernel):
            out += padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    return (out / float(kernel * kernel)).astype(arr.dtype, copy=False)


def find_field_min_center(
    msl: np.ndarray,
    true_lat: float,
    true_lon: float,
    domain: DomainConfig,
    search_radius_km: float,
    *,
    smooth_px: int = 0,
) -> tuple[float, float]:
    """Find the local pressure-basin minimum containing the IBTrACS position.

    The previous implementation selected the global MSL minimum inside a large
    disk around the best-track position. In multi-low scenes that can snap the
    reference to a deeper but unrelated cyclone. This routine instead starts at
    the nearest grid point to the best-track position and follows the steepest
    8-neighbor descent until a local minimum is reached, with
    ``search_radius_km`` acting only as a safety cap on displacement from the
    true position.
    """

    if msl.shape != domain.shape:
        raise ValueError(f"msl shape {msl.shape} does not match domain {domain.shape}")

    field = _box_smooth(msl, int(smooth_px)) if smooth_px > 0 else msl
    height, width = domain.shape
    y_f, x_f = latlon_to_grid(true_lat, true_lon, domain, clip=True)
    y = int(np.clip(round(float(y_f)), 0, height - 1))
    x = int(np.clip(round(float(x_f)), 0, width - 1))

    while True:
        best_y, best_x = y, x
        best_val = float(field[y, x])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    value = float(field[ny, nx])
                    if value < best_val:
                        best_y, best_x, best_val = ny, nx, value
        if (best_y, best_x) == (y, x):
            break
        cand_lat, cand_lon = grid_to_latlon(best_y, best_x, domain)
        if float(haversine_km(cand_lat, cand_lon, true_lat, true_lon)) > search_radius_km:
            break
        y, x = best_y, best_x

    lat, lon = grid_to_latlon(y, x, domain)
    return float(lat), float(lon)


def _iter_record_dicts(records: pd.DataFrame | Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    """Yield records as dictionaries."""

    if isinstance(records, pd.DataFrame):
        yield from records.to_dict("records")
    else:
        yield from records


def generate_labels(
    *,
    msl: np.ndarray,
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    domain: DomainConfig,
    label_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Generate heatmap, offset, and mask arrays for one field time."""

    mode = label_config.get("mode")
    if mode not in {"ibtracs", "in_field"}:
        raise ValueError("labels.mode must be 'ibtracs' or 'in_field'")
    sigma = float(label_config.get("sigma_px", 3.0))
    search_radius_km = float(label_config.get("search_radius_km", 300.0))
    smooth_px = int(label_config.get("field_center_smooth_px", 0))

    heatmap = np.zeros(domain.shape, dtype=np.float32)
    offset = np.zeros((2, domain.height, domain.width), dtype=np.float32)
    mask = np.zeros(domain.shape, dtype=np.uint8)

    yy, xx = np.indices(domain.shape, dtype=np.float32)
    for record in _iter_record_dicts(records):
        lat_true = float(record["LAT"])
        lon_true = float(record["LON"])
        if not in_domain(lat_true, lon_true, domain):
            continue
        if mode == "in_field":
            center_lat, center_lon = find_field_min_center(
                msl,
                lat_true,
                lon_true,
                domain,
                search_radius_km,
                smooth_px=smooth_px,
            )
        else:
            center_lat, center_lon = lat_true, lon_true

        cy_f, cx_f = latlon_to_grid(center_lat, center_lon, domain, clip=True)
        cy = float(cy_f)
        cx = float(cx_f)
        gaussian = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))).astype(np.float32)
        heatmap = np.maximum(heatmap, gaussian)

        iy = int(np.floor(cy))
        ix = int(np.floor(cx))
        iy = int(np.clip(iy, 0, domain.height - 1))
        ix = int(np.clip(ix, 0, domain.width - 1))
        heatmap[iy, ix] = 1.0
        offset[0, iy, ix] = cy - float(np.floor(cy))
        offset[1, iy, ix] = cx - float(np.floor(cx))
        mask[iy, ix] = 1

    return {"heatmap": heatmap, "offset": offset, "mask": mask}


def save_label_npz(label: Mapping[str, np.ndarray], path: str | Path) -> None:
    """Save a label dictionary as a compressed npz file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, heatmap=label["heatmap"], offset=label["offset"], mask=label["mask"])


def load_label_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a label npz file."""

    with np.load(path) as data:
        return {
            "heatmap": data["heatmap"].astype(np.float32),
            "offset": data["offset"].astype(np.float32),
            "mask": data["mask"].astype(np.uint8),
        }
