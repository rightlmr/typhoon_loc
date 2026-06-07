"""ERA5 pretraining entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
import torch
from torch.utils.data import DataLoader, random_split

from tclocator.common import PHASE0_REQUIRED_MESSAGE, iter_files, load_config, resolve_device, set_seed
from tclocator.dataset import FieldDataset, SyntheticTCDataset, build_era5_samples, collate_batch
from tclocator.decode import decode_heatmap
from tclocator.labels import read_ibtracs
from tclocator.losses import LossConfig, TCLocatorLoss
from tclocator.model import build_model_from_config
from tclocator.normalization import load_norm_stats


def _phase0_gate(config: dict[str, Any]) -> bool:
    """Require Phase 0 label mode before training."""

    if config.get("labels", {}).get("mode") is None:
        print(PHASE0_REQUIRED_MESSAGE)
        return False
    return True


def _center_mae_km(model: torch.nn.Module, loader: DataLoader, device: str, config: dict[str, Any]) -> float:
    """Decode validation predictions and match them to positive label centers.

    A field can contain multiple active cyclones. Validation therefore cannot
    compare the highest-confidence prediction with the first mask pixel only;
    doing so turns a correct detection of a different cyclone into a false
    multi-thousand-kilometer error. This metric greedily matches decoded peaks
    to all label centers inside the configured decode latitude range.
    """

    from tclocator.common import DomainConfig, grid_to_latlon, haversine_km

    domain = DomainConfig.from_mapping(config.get("domain"))
    lat_filter = tuple(config.get("decode", {}).get("lat_filter", [0.0, 40.0]))
    model.eval()
    errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            outputs = model(inputs)
            for i in range(inputs.shape[0]):
                ys, xs = torch.where(batch["mask"][i] > 0.5)
                refs: list[tuple[float, float]] = []
                for y_idx, x_idx in zip(ys.tolist(), xs.tolist(), strict=True):
                    y = float(y_idx) + float(batch["offset"][i, 0, y_idx, x_idx])
                    x = float(x_idx) + float(batch["offset"][i, 1, y_idx, x_idx])
                    lat_ref, lon_ref = grid_to_latlon(y, x, domain)
                    if lat_filter[0] <= float(lat_ref) <= lat_filter[1]:
                        refs.append((float(lat_ref), float(lon_ref)))
                if not refs:
                    continue

                decoded = decode_heatmap(
                    outputs["heatmap"][i, 0],
                    outputs["offset"][i],
                    domain,
                    conf_thresh=0.01,
                    lat_filter=lat_filter,
                    topk=max(20, len(refs) * 5),
                )
                if decoded.empty:
                    continue

                remaining = set(range(len(decoded)))
                for lat_ref, lon_ref in refs:
                    best_idx: int | None = None
                    best_dist = float("inf")
                    for pred_idx in list(remaining):
                        row = decoded.iloc[pred_idx]
                        dist = float(haversine_km(float(row["LAT"]), float(row["LON"]), lat_ref, lon_ref))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = pred_idx
                    if best_idx is not None:
                        remaining.remove(best_idx)
                        errors.append(best_dist)
    return float(sum(errors) / max(len(errors), 1)) if errors else float("inf")


def _train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: dict[str, Any], device: str) -> dict[str, Any]:
    """Train a model and return the best checkpoint payload."""

    train_cfg = config.get("train", {})
    loss_fn = TCLocatorLoss(LossConfig.from_config(config))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    patience = int(train_cfg.get("patience", 6))
    best_mae = float("inf")
    best_payload: dict[str, Any] | None = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model(batch["input"]), batch)
            losses["loss"].backward()
            optimizer.step()
            running += float(losses["loss"].detach().cpu())
        scheduler.step()
        val_mae = _center_mae_km(model, val_loader, device, config)
        print(f"epoch={epoch + 1} train_loss={running / max(len(train_loader), 1):.4f} val_center_mae_km={val_mae:.2f}")
        if val_mae < best_mae:
            best_mae = val_mae
            best_payload = model.checkpoint_payload(config)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_payload or model.checkpoint_payload(config)


def _real_dataset(config: dict[str, Any]) -> FieldDataset | None:
    """Build the real ERA5 dataset when data exists."""

    files = iter_files(config.get("paths", {}).get("era5_dir", ""), [".nc"])
    if not files:
        print("No ERA5 files found; pretraining skipped.")
        return None
    norm_path = Path(config.get("paths", {}).get("norm_stats_era5", ""))
    norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None
    ib_path = Path(config.get("paths", {}).get("ibtracs_csv", ""))
    records = read_ibtracs(ib_path, config.get("ibtracs", {}).get("col_map", {})) if ib_path.exists() else None
    samples = build_era5_samples(files, config=config)
    return FieldDataset(samples=samples, config=config, norm_stats=norm_stats, ibtracs_records=records)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pretrain.yaml"))
    parser.add_argument("--smoke-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not _phase0_gate(config):
        return 1
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    model = build_model_from_config(config).to(device)

    if args.smoke_synthetic:
        dataset = SyntheticTCDataset(length=4, channels=config["channels"], seed=int(config.get("seed", 42)))
    else:
        dataset = _real_dataset(config)
        if dataset is None:
            return 0
    val_size = max(1, len(dataset) // 4)
    train_size = max(1, len(dataset) - val_size)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(int(config.get("seed", 42))))
    loader_cfg = config.get("train", {})
    train_loader = DataLoader(train_ds, batch_size=int(loader_cfg.get("batch_size", 2)), shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_batch)
    payload = _train(model, train_loader, val_loader, config, device)
    ckpt_path = Path(config.get("paths", {}).get("pretrain_ckpt", ROOT / "outputs" / "pretrain_best.ckpt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)
    print(f"Wrote {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
