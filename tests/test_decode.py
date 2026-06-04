"""Decode unit tests."""

from __future__ import annotations

import numpy as np

from tclocator.common import DomainConfig, latlon_to_grid
from tclocator.decode import decode_heatmap


def test_decode_gaussian_peak_error_below_one_pixel() -> None:
    """Synthetic Gaussian heatmap decodes within one pixel."""

    domain = DomainConfig(lat_min=0.0, lat_max=15.75, lon_min=100.0, lon_max=123.75, res=0.25)
    cy, cx = 18.35, 42.65
    yy, xx = np.indices(domain.shape, dtype=np.float32)
    heatmap = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 2.0**2))).astype(np.float32)
    py = int(round(cy))
    px = int(round(cx))
    heatmap[py, px] = 1.0
    offset = np.zeros((2, domain.height, domain.width), dtype=np.float32)
    offset[0, py, px] = cy - py
    offset[1, py, px] = cx - px

    decoded = decode_heatmap(heatmap, offset, domain, conf_thresh=0.5, lat_filter=(0.0, 40.0), topk=1)

    assert len(decoded) == 1
    pred_y, pred_x = latlon_to_grid(float(decoded.iloc[0]["LAT"]), float(decoded.iloc[0]["LON"]), domain)
    assert abs(float(pred_y) - cy) < 1.0
    assert abs(float(pred_x) - cx) < 1.0

