"""Check old global-argmin centers against bounded local-descent centers."""

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

from tclocator.common import DomainConfig, build_lat_lon, haversine_km, iter_files, load_config
from tclocator.labels import find_field_min_center, read_ibtracs, records_at_time


def _old_global_argmin_center(
    msl: np.ndarray,
    true_lat: float,
    true_lon: float,
    domain: DomainConfig,
    radius_km: float,
) -> tuple[float, float]:
    """Replicate the former disk-wide global-argmin behavior for comparison."""

    lat1d, lon1d = build_lat_lon(domain)
    dist = haversine_km(lat1d[:, None], lon1d[None, :], true_lat, true_lon)
    mask = dist <= radius_km
    if not np.any(mask):
        return true_lat, float(np.mod(true_lon, 360.0))
    masked = np.where(mask, msl, np.inf)
    y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return float(lat1d[y]), float(lon1d[x])


def _collect_distances(
    config: dict[str, Any],
    max_samples: int,
    old_radius_km: float,
    *,
    new_radius_km: float | None = None,
) -> tuple[list[float], list[float]]:
    """Collect old/new distances to IBTrACS truth from short-lead AIFS fields."""

    paths = config.get("paths", {})
    domain = DomainConfig.from_mapping(config.get("domain"))
    label_cfg = config.get("labels", {})
    radius_km = float(new_radius_km if new_radius_km is not None else label_cfg.get("search_radius_km", 100.0))
    smooth_px = int(label_cfg.get("field_center_smooth_px", 0))
    lead_max = int(config.get("finetune", {}).get("lead_max", 24))
    files = iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
    records = read_ibtracs(paths.get("ibtracs_csv", ""), config.get("ibtracs", {}).get("col_map", {}))
    old_distances: list[float] = []
    new_distances: list[float] = []

    for path in files:
        meta = parse_aifs_filename(path)
        if meta.forecast_hour > lead_max:
            continue
        at_time = records_at_time(records, pd.Timestamp(meta.valid_time))
        if at_time.empty:
            continue
        field, _ = read_aifs_channels(path, channels=["msl"], domain=domain, aifs_config=config.get("aifs", {}))
        for _, record in at_time.iterrows():
            true_lat = float(record["LAT"])
            true_lon = float(record["LON"])
            old_lat, old_lon = _old_global_argmin_center(field[0], true_lat, true_lon, domain, old_radius_km)
            new_lat, new_lon = find_field_min_center(
                field[0],
                true_lat,
                true_lon,
                domain,
                radius_km,
                smooth_px=smooth_px,
            )
            old_distances.append(float(haversine_km(true_lat, true_lon, old_lat, old_lon)))
            new_distances.append(float(haversine_km(true_lat, true_lon, new_lat, new_lon)))
            if len(new_distances) >= max_samples:
                return old_distances, new_distances
    return old_distances, new_distances


def _scan_radii(config: dict[str, Any], max_samples: int, radii: list[float]) -> None:
    """Print median/p90 true-distance for several descent caps."""

    for radius in radii:
        _, distances = _collect_distances(config, max_samples, old_radius_km=500.0, new_radius_km=radius)
        if not distances:
            print(f"radius={radius:g}: no samples")
            continue
        median = float(np.median(distances))
        p90 = float(np.quantile(distances, 0.90))
        lt100 = sum(value < 100.0 for value in distances)
        print(f"radius={radius:g}: median={median:.2f} km p90={p90:.2f} km lt100={lt100}/{len(distances)}")


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--old-radius-km", type=float, default=500.0)
    parser.add_argument("--scan-radii", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.scan_radii:
        _scan_radii(config, args.max_samples, [50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0])
        return 0

    old_distances, new_distances = _collect_distances(config, args.max_samples, args.old_radius_km)
    if not new_distances:
        print("No matching short-lead AIFS/IBTrACS samples found.")
        return 1

    old_median = float(np.median(old_distances))
    new_median = float(np.median(new_distances))
    n = len(new_distances)
    print(f"old_global_argmin: median_dist_to_truth={old_median:.2f} km (n={n})")
    print(f"new_local_descent: median_dist_to_truth={new_median:.2f} km (n={n})")
    passed = new_median < 100.0 and new_median < old_median * 0.5
    print("PASS" if passed else "FAIL")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
