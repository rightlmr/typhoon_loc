"""AIFS GRIB2 reading, filename parsing, domain cropping, and channel stacking."""

from __future__ import annotations

try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path.*")
        import pygrib  # type: ignore
except ImportError:  # pragma: no cover - exercised only when pygrib is absent
    pygrib = None  # type: ignore

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from tclocator.common import DomainConfig, build_lat_lon, crop_regular_latlon_grid
from tclocator.vorticity import calc_vo850


GLOBAL_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float64)
GLOBAL_LON = np.linspace(0.0, 359.75, 1440, dtype=np.float64)


@dataclass(frozen=True)
class AIFSFileMeta:
    """Metadata parsed from an AIFS forecast file name."""

    init_time: datetime
    forecast_hour: int
    valid_time: datetime


GRIB_KEYS: dict[str, tuple[str, str, int]] = {
    "msl": ("msl", "meanSea", 0),
    "mslp": ("msl", "meanSea", 0),
    "u10": ("10u", "heightAboveGround", 10),
    "v10": ("10v", "heightAboveGround", 10),
    "t2": ("2t", "heightAboveGround", 2),
    "u850": ("u", "isobaricInhPa", 850),
    "v850": ("v", "isobaricInhPa", 850),
    "q850": ("q", "isobaricInhPa", 850),
    "t850": ("t", "isobaricInhPa", 850),
    "t_850": ("t", "isobaricInhPa", 850),
    "u700": ("u", "isobaricInhPa", 700),
    "v700": ("v", "isobaricInhPa", 700),
    "q700": ("q", "isobaricInhPa", 700),
    "t700": ("t", "isobaricInhPa", 700),
    "t_700": ("t", "isobaricInhPa", 700),
    "u500": ("u", "isobaricInhPa", 500),
    "v500": ("v", "isobaricInhPa", 500),
    "q500": ("q", "isobaricInhPa", 500),
    "t500": ("t", "isobaricInhPa", 500),
    "t_500": ("t", "isobaricInhPa", 500),
}


def parse_aifs_filename(path: str | Path) -> AIFSFileMeta:
    """Parse supported AIFS file names into forecast metadata."""

    name = Path(path).name
    m1 = re.fullmatch(r"AIFS_(\d{4})_(\d{2})_(\d{2})_(\d{2})_FCST_(\d+)h\.grib2", name)
    if m1:
        year, month, day, hour, lead = m1.groups()
        init_time = datetime(int(year), int(month), int(day), int(hour), tzinfo=timezone.utc)
        forecast_hour = int(lead)
        return AIFSFileMeta(init_time, forecast_hour, init_time + timedelta(hours=forecast_hour))

    m2 = re.fullmatch(r"(\d{14})-(\d+)h-oper-fc\.grib2", name)
    if m2:
        init_raw, lead = m2.groups()
        init_time = datetime.strptime(init_raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        forecast_hour = int(lead)
        return AIFSFileMeta(init_time, forecast_hour, init_time + timedelta(hours=forecast_hour))

    raise ValueError(f"Unsupported AIFS filename format: {name}")


def _require_pygrib() -> Any:
    """Return pygrib or raise a dependency error."""

    if pygrib is None:
        raise ImportError("pygrib is required to read AIFS GRIB2 files")
    return pygrib


def read_aifs_variable(path: str | Path, internal_name: str) -> np.ndarray:
    """Read one configured variable from an AIFS GRIB2 file."""

    key = internal_name.replace("_", "") if internal_name not in GRIB_KEYS else internal_name
    if key not in GRIB_KEYS:
        raise KeyError(f"Unsupported AIFS internal variable: {internal_name}")
    short_name, type_of_level, level = GRIB_KEYS[key]
    grib = _require_pygrib()
    with grib.open(str(path)) as grbs:
        selected = grbs.select(shortName=short_name, typeOfLevel=type_of_level, level=level)
        if not selected:
            raise KeyError(f"GRIB variable not found: {GRIB_KEYS[key]} in {path}")
        return np.asarray(selected[0].values, dtype=np.float32)


def crop_aifs_global(values: np.ndarray, domain: DomainConfig) -> np.ndarray:
    """Crop an AIFS global 721x1440 field to the configured domain."""

    return crop_regular_latlon_grid(values, GLOBAL_LAT, GLOBAL_LON, domain)


def read_aifs_channels(
    path: str | Path,
    *,
    channels: Sequence[str],
    domain: DomainConfig,
    aifs_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a configured AIFS field as ``[C,H,W]`` float32."""

    _ = aifs_config
    arrays: dict[str, np.ndarray] = {}

    def read_crop(name: str) -> np.ndarray:
        arr = arrays.get(name)
        if arr is None:
            arr = crop_aifs_global(read_aifs_variable(path, name), domain)
            arrays[name] = arr
        return arr

    for channel in channels:
        if channel == "vo_850":
            u = read_crop("u850")
            v = read_crop("v850")
            lat1d, lon1d = build_lat_lon(domain)
            arrays[channel] = calc_vo850(u, v, lat1d, lon1d)
        else:
            arrays[channel] = read_crop(channel)

    meta = parse_aifs_filename(path)
    stacked = np.stack([arrays[channel] for channel in channels], axis=0).astype(np.float32)
    return stacked, {
        "path": str(path),
        "init_time": meta.init_time.isoformat(),
        "forecast_hour": meta.forecast_hour,
        "valid_time": meta.valid_time.isoformat(),
    }
