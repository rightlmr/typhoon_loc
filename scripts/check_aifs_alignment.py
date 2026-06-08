"""Gate AIFS .pt spatial alignment against known strong lead-0 storms."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
import numpy as np

from tclocator.common import DomainConfig, build_lat_lon, grid_to_latlon, haversine_km, iter_files, latlon_to_grid, load_config


@dataclass(frozen=True)
class AlignmentCase:
    """One known strong storm alignment check case."""

    valid_time: str
    lat: float
    lon: float
    sid: str = ""


DEFAULT_CASES = [
    AlignmentCase(valid_time="2024-09-07T12:00:00", lat=21.0, lon=106.0, sid="2024244N09137"),
]


def _find_lead0_file(config: dict[str, Any], valid_time: str) -> Path:
    """Find a lead-0 AIFS file by valid time."""

    target = np.datetime64(valid_time)
    files = iter_files(config.get("paths", {}).get("aifs_dir", ""), [".pt", ".grib2", ".grb2", ".grib"])
    for path in files:
        meta = parse_aifs_filename(path)
        if meta.forecast_hour == 0 and np.datetime64(meta.valid_time.replace(tzinfo=None)) == target:
            return path
    raise FileNotFoundError(f"No lead-0 AIFS file found for valid_time={valid_time}")


def _truth_value(msl: np.ndarray, lat: float, lon: float, domain: DomainConfig) -> float:
    """Return nearest-grid MSL at the truth location."""

    y_f, x_f = latlon_to_grid(lat, lon, domain, clip=True)
    y = int(np.clip(round(float(y_f)), 0, domain.height - 1))
    x = int(np.clip(round(float(x_f)), 0, domain.width - 1))
    return float(msl[y, x])


def _local_min(msl: np.ndarray, lat: float, lon: float, domain: DomainConfig, radius_km: float) -> tuple[float, float, float, float]:
    """Return local MSL minimum and distance to truth."""

    lat1d, lon1d = build_lat_lon(domain)
    dist = haversine_km(lat1d[:, None], lon1d[None, :], lat, lon)
    mask = dist <= radius_km
    if not np.any(mask):
        raise ValueError("No grid points inside alignment radius")
    masked = np.where(mask, msl, np.inf)
    y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
    min_lat, min_lon = grid_to_latlon(float(y), float(x), domain)
    return float(masked[y, x]), float(min_lat), float(min_lon), float(dist[y, x])


def _parse_case(raw: str) -> AlignmentCase:
    """Parse a CLI case as valid_time,lat,lon[,sid]."""

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("case must be valid_time,lat,lon[,sid]")
    return AlignmentCase(valid_time=parts[0], lat=float(parts[1]), lon=float(parts[2]), sid=parts[3] if len(parts) == 4 else "")


def _check_case(config: dict[str, Any], domain: DomainConfig, case: AlignmentCase) -> tuple[bool, str]:
    """Run one alignment case through the production reader."""

    path = _find_lead0_file(config, case.valid_time)
    field, _ = read_aifs_channels(path, channels=["msl"], domain=domain, aifs_config=config.get("aifs", {}))
    msl = field[0]
    truth = _truth_value(msl, case.lat, case.lon, domain)
    local_min, min_lat, min_lon, min_dist = _local_min(msl, case.lat, case.lon, domain, radius_km=100.0)
    ok = truth < 99_500.0 and local_min < 99_000.0
    label = case.sid or Path(path).stem
    message = (
        f"{label} valid={case.valid_time} truth=({case.lat:.2f},{case.lon:.2f}) "
        f"msl_truth_hPa={truth / 100.0:.2f} "
        f"min100_hPa={local_min / 100.0:.2f} min100=({min_lat:.2f},{min_lon:.2f}) "
        f"min100_dist_km={min_dist:.2f} {'PASS' if ok else 'FAIL'}"
    )
    return ok, message


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--case", action="append", type=_parse_case, default=None, help="valid_time,lat,lon[,sid]")
    args = parser.parse_args()

    config = load_config(args.config)
    domain = DomainConfig.from_mapping(config.get("domain"))
    cases = list(args.case) if args.case else DEFAULT_CASES
    results = [_check_case(config, domain, case) for case in cases]
    for _, message in results:
        print(message)
    passed = all(ok for ok, _ in results)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
