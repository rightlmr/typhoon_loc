"""M1 synthetic overfit test."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tclocator.common import DomainConfig, grid_to_latlon, haversine_km, latlon_to_grid, set_seed
from tclocator.dataset import SyntheticTCDataset
from tclocator.decode import decode_heatmap
from tclocator.losses import LossConfig, TCLocatorLoss
from tclocator.model import TCLocatorUNet


def test_synthetic_era5_overfit_loc_error_below_30_km() -> None:
    """A tiny synthetic ERA5-like sample can be overfit below 30 km."""

    set_seed(7)
    domain = DomainConfig(lat_min=0.0, lat_max=7.75, lon_min=100.0, lon_max=115.75, res=0.25)
    center = grid_to_latlon(14.25, 31.5, domain)
    dataset = SyntheticTCDataset(length=1, channels=("msl", "vo_850", "t_500"), domain=domain, fixed_center=center)
    sample = dataset[0]
    x = sample["input"].unsqueeze(0).float()
    x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    batch = {
        "input": x,
        "heatmap": sample["heatmap"].unsqueeze(0).float(),
        "offset": sample["offset"].unsqueeze(0).float(),
        "mask": sample["mask"].unsqueeze(0).float(),
    }

    model = TCLocatorUNet(in_channels=3, base_channels=4)
    loss_fn = TCLocatorLoss(LossConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(model(batch["input"]), batch)
        losses["loss"].backward()
        optimizer.step()

    with torch.no_grad():
        outputs = model(batch["input"])
    decoded = decode_heatmap(
        outputs["heatmap"][0, 0],
        outputs["offset"][0],
        domain,
        conf_thresh=0.05,
        lat_filter=(0.0, 40.0),
        topk=1,
    )
    assert not decoded.empty
    cy, cx = latlon_to_grid(center[0], center[1], domain)
    y_pred, x_pred = latlon_to_grid(float(decoded.iloc[0]["LAT"]), float(decoded.iloc[0]["LON"]), domain)
    pixel_error_km = float(haversine_km(center[0], center[1], decoded.iloc[0]["LAT"], decoded.iloc[0]["LON"]))
    assert abs(float(y_pred) - float(cy)) < 1.2
    assert abs(float(x_pred) - float(cx)) < 1.2
    assert pixel_error_km < 30.0

