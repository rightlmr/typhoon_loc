"""IBTrACS-driven field-consistent label generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from tclocator.common import DomainConfig, build_lat_lon, haversine_km, in_domain, latlon_to_grid


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


def find_field_min_center(
    msl: np.ndarray,
    true_lat: float,
    true_lon: float,
    domain: DomainConfig,
    search_radius_km: float,
) -> tuple[float, float]:
    """Find the minimum sea-level pressure center near an IBTrACS position."""

    if msl.shape != domain.shape:
        raise ValueError(f"msl shape {msl.shape} does not match domain {domain.shape}")
    lat1d, lon1d = build_lat_lon(domain)
    lat2d = lat1d[:, None]
    lon2d = lon1d[None, :]
    dist = haversine_km(lat2d, lon2d, true_lat, true_lon)
    mask = dist <= search_radius_km
    if not np.any(mask):
        return true_lat, float(np.mod(true_lon, 360.0))
    masked = np.where(mask, msl, np.inf)
    y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return float(lat1d[y]), float(lon1d[x])


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
            center_lat, center_lon = find_field_min_center(msl, lat_true, lon_true, domain, search_radius_km)
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

