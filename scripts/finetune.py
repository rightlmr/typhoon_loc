"""AIFS fine-tuning entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
import torch
from torch.utils.data import DataLoader, Subset

from tclocator.common import PHASE0_REQUIRED_MESSAGE, iter_files, load_config, resolve_device, set_seed
from tclocator.dataset import FieldDataset, FieldSample, SyntheticTCDataset, build_aifs_samples, collate_batch
from tclocator.losses import LossConfig, TCLocatorLoss
from tclocator.model import build_model_from_config
from tclocator.normalization import load_norm_stats
from tclocator.split import filter_aifs_files_by_usable_months, grouped_split, split_config, split_val_groups_override
from scripts.pretrain import _center_mae_km


def _phase0_gate(config: dict[str, Any]) -> bool:
    """Require Phase 0 label mode and lead limit before fine-tuning."""

    if config.get("labels", {}).get("mode") is None or config.get("finetune", {}).get("lead_max") is None:
        print(PHASE0_REQUIRED_MESSAGE)
        return False
    return True


def _load_pretrained(model: torch.nn.Module, config: dict[str, Any]) -> None:
    """Load ERA5 pretraining weights when available."""

    ckpt_path = Path(config.get("paths", {}).get("pretrain_ckpt", ""))
    if not ckpt_path.exists():
        print(f"Pretrain checkpoint not found: {ckpt_path}; starting fine-tune model from current initialization.")
        return
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    print(f"Loaded {ckpt_path}")


def _train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: dict[str, Any], device: str) -> dict[str, Any]:
    """Fine-tune decoder and heads."""

    train_cfg = config.get("train", {})
    loss_fn = TCLocatorLoss(LossConfig.from_config(config))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 3e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 12))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    patience = int(train_cfg.get("patience", 4))
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


def _real_dataset(config: dict[str, Any]) -> tuple[FieldDataset, list[FieldSample]] | None:
    """Build real AIFS fine-tuning dataset."""

    files = filter_aifs_files_by_usable_months(
        config,
        iter_files(config.get("paths", {}).get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"]),
    )
    if not files:
        print("No AIFS files found; fine-tuning skipped.")
        return None
    lead_max = int(config.get("finetune", {}).get("lead_max"))
    samples = build_aifs_samples(files, lead_max=lead_max, config=config)
    if not samples:
        print("No AIFS samples remain after lead_max filtering; fine-tuning skipped.")
        return None
    norm_path = Path(config.get("paths", {}).get("norm_stats_aifs", ""))
    norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None
    dataset = FieldDataset(samples=samples, config=config, norm_stats=norm_stats, ibtracs_records=None)
    return dataset, samples


def _smoke_split(dataset: SyntheticTCDataset) -> tuple[Subset, Subset]:
    """Return a deterministic split for synthetic smoke tests."""

    val_size = max(1, len(dataset) // 4)
    train_size = max(1, len(dataset) - val_size)
    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _write_aifs_split_record(config: dict[str, Any], group_by: str, val_fraction: float, seed: int, val_groups: list[str]) -> None:
    """Write the deterministic AIFS validation groups for transparency."""

    out_dir = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"group_by": group_by, "val_fraction": val_fraction, "seed": seed, "val_groups": val_groups}
    (out_dir / "split_aifs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--smoke-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not _phase0_gate(config):
        return 1
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    model = build_model_from_config(config).to(device)
    _load_pretrained(model, config)
    if bool(config.get("finetune", {}).get("freeze_encoder", True)):
        model.freeze_encoder()

    if args.smoke_synthetic:
        dataset = SyntheticTCDataset(length=4, channels=config["channels"], seed=int(config.get("seed", 42)))
        train_ds, val_ds = _smoke_split(dataset)
    else:
        real = _real_dataset(config)
        if real is None:
            return 0
        dataset, samples = real
        group_by, val_fraction, split_seed = split_config(config, "init_time")
        train_idx, val_idx, val_groups = grouped_split(
            samples,
            group_by=group_by,
            val_fraction=val_fraction,
            seed=split_seed,
            val_groups_override=split_val_groups_override(config),
        )
        _write_aifs_split_record(config, group_by, val_fraction, split_seed, val_groups)
        if not train_idx or not val_idx:
            print(f"Invalid AIFS split: train n={len(train_idx)} val n={len(val_idx)} val_groups={val_groups}")
            return 1
        print(f"AIFS split group_by={group_by} train n={len(train_idx)} val n={len(val_idx)} val_groups={val_groups}")
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

    loader_cfg = config.get("train", {})
    train_loader = DataLoader(train_ds, batch_size=int(loader_cfg.get("batch_size", 1)), shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_batch)
    payload = _train(model, train_loader, val_loader, config, device)
    ckpt_path = Path(config.get("paths", {}).get("finetune_ckpt", ROOT / "outputs" / "finetune_best.ckpt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)
    print(f"Wrote {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
