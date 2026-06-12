"""Label center selection tests."""

from __future__ import annotations

import numpy as np

from tclocator.common import DomainConfig, build_lat_lon, haversine_km, latlon_to_grid
from tclocator.labels import find_field_min_center, find_hybrid_center


def _old_global_argmin_center(
    msl: np.ndarray,
    true_lat: float,
    true_lon: float,
    domain: DomainConfig,
    radius_km: float,
) -> tuple[float, float]:
    """Former disk-wide argmin implementation used only for regression proof."""

    lat1d, lon1d = build_lat_lon(domain)
    dist = haversine_km(lat1d[:, None], lon1d[None, :], true_lat, true_lon)
    masked = np.where(dist <= radius_km, msl, np.inf)
    y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return float(lat1d[y]), float(lon1d[x])


def _two_low_field(domain: DomainConfig, target: tuple[float, float], distractor: tuple[float, float]) -> np.ndarray:
    """Create a pressure field with a shallow target low and a deeper distractor low."""

    yy, xx = np.indices(domain.shape, dtype=np.float32)
    target_y, target_x = latlon_to_grid(target[0], target[1], domain)
    distractor_y, distractor_x = latlon_to_grid(distractor[0], distractor[1], domain)
    target_r2 = (yy - float(target_y)) ** 2 + (xx - float(target_x)) ** 2
    distractor_r2 = (yy - float(distractor_y)) ** 2 + (xx - float(distractor_x)) ** 2
    msl = 101000.0 - 350.0 * np.exp(-target_r2 / (2.0 * 2.0**2))
    msl -= 1300.0 * np.exp(-distractor_r2 / (2.0 * 2.0**2))
    return msl.astype(np.float32)


def test_field_center_stays_in_true_pressure_basin() -> None:
    """Local descent stays on the target basin while old global argmin snaps away."""

    domain = DomainConfig(lat_min=0.0, lat_max=10.0, lon_min=100.0, lon_max=112.0, res=0.25)
    target = (5.0, 105.0)
    distractor = (5.0, 108.2)
    msl = _two_low_field(domain, target, distractor)

    new_center = find_field_min_center(msl, target[0], target[1], domain, 250.0)
    old_center = _old_global_argmin_center(msl, target[0], target[1], domain, 500.0)

    assert float(haversine_km(new_center[0], new_center[1], target[0], target[1])) < 40.0
    assert float(haversine_km(old_center[0], old_center[1], distractor[0], distractor[1])) < 40.0


def test_field_center_cap_prevents_long_descent() -> None:
    """The descent cap prevents a broad slope from walking far from truth."""

    domain = DomainConfig(lat_min=0.0, lat_max=10.0, lon_min=100.0, lon_max=112.0, res=0.25)
    true_lat, true_lon = 5.0, 105.0
    yy, xx = np.indices(domain.shape, dtype=np.float32)
    far_y, far_x = latlon_to_grid(5.0, 110.0, domain)
    msl = 101000.0 - 2000.0 * np.exp(-(((yy - float(far_y)) ** 2 + (xx - float(far_x)) ** 2) / (2.0 * 12.0**2)))

    center = find_field_min_center(msl.astype(np.float32), true_lat, true_lon, domain, 150.0)

    assert float(haversine_km(center[0], center[1], true_lat, true_lon)) <= 155.0


def test_hybrid_uses_msl_when_descent_reaches_local_min() -> None:
    """Hybrid keeps the MSL center when descent does not hit the cap."""

    domain = DomainConfig(lat_min=0.0, lat_max=10.0, lon_min=100.0, lon_max=112.0, res=0.25)
    target = (5.0, 105.0)
    distractor = (5.0, 108.0)
    msl = _two_low_field(domain, target, distractor)
    vo = np.zeros(domain.shape, dtype=np.float32)
    y, x = latlon_to_grid(distractor[0], distractor[1], domain)
    vo[int(round(float(y))), int(round(float(x)))] = 1.0

    center_lat, center_lon, source = find_hybrid_center(msl, vo, target[0], target[1], domain, 250.0)

    assert source == "msl"
    assert float(haversine_km(center_lat, center_lon, target[0], target[1])) < 40.0


def test_hybrid_uses_vo_when_msl_descent_hits_cap() -> None:
    """Hybrid switches to the strongest local vo_850 point only on cap stops."""

    domain = DomainConfig(lat_min=0.0, lat_max=10.0, lon_min=100.0, lon_max=112.0, res=0.25)
    true_lat, true_lon = 5.0, 105.0
    vo_lat, vo_lon = 5.0, 105.5
    yy, xx = np.indices(domain.shape, dtype=np.float32)
    far_y, far_x = latlon_to_grid(5.0, 110.0, domain)
    msl = 101000.0 - 2000.0 * np.exp(-(((yy - float(far_y)) ** 2 + (xx - float(far_x)) ** 2) / (2.0 * 12.0**2)))
    vo = np.zeros(domain.shape, dtype=np.float32)
    vy, vx = latlon_to_grid(vo_lat, vo_lon, domain)
    vo[int(round(float(vy))), int(round(float(vx)))] = 1.0

    center_lat, center_lon, source = find_hybrid_center(msl.astype(np.float32), vo, true_lat, true_lon, domain, 80.0, vo_smooth_px=0)

    assert source == "vo"
    assert float(haversine_km(center_lat, center_lon, vo_lat, vo_lon)) < 1.0
