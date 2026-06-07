"""Compute per-domain normalization statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tclocator.common import DomainConfig, iter_files, load_config
from tclocator.dataset import SyntheticTCDataset
from tclocator.io_aifs import read_aifs_channels
from tclocator.io_era5 import read_era5_channels
from tclocator.normalization import compute_norm_stats_stream, save_norm_stats


def _synthetic_samples(config: dict[str, Any]) -> list[np.ndarray]:
    """Return synthetic samples for smoke testing."""

    dataset = SyntheticTCDataset(length=4, channels=config["channels"], seed=int(config.get("seed", 42)))
    return [dataset[i]["input"].numpy() for i in range(len(dataset))]


def _compute_era5(config: dict[str, Any], *, smoke: bool) -> None:
    """Compute ERA5 stats."""

    paths = config.get("paths", {})
    domain = DomainConfig.from_mapping(config.get("domain"))
    if smoke:
        samples = _synthetic_samples(config)
        factory = lambda: iter(samples)
    else:
        files = iter_files(paths.get("era5_dir", ""), [".nc"])
        if not files:
            print("No ERA5 files found; skipping ERA5 normalization stats.")
            return

        def factory() -> Any:
            for path in files:
                yield read_era5_channels(path, channels=config["channels"], domain=domain, era5_config=config.get("era5", {}))[0]

    stats = compute_norm_stats_stream(factory, config["channels"], config.get("norm", {}))
    out = Path(paths.get("norm_stats_era5", ROOT / "outputs" / "norm_stats_era5.json"))
    save_norm_stats(stats, out)
    print(f"Wrote {out}")


def _compute_aifs(config: dict[str, Any], *, smoke: bool) -> None:
    """Compute AIFS stats."""

    paths = config.get("paths", {})
    domain = DomainConfig.from_mapping(config.get("domain"))
    if smoke:
        samples = _synthetic_samples(config)
        factory = lambda: iter(samples)
    else:
        files = iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
        if not files:
            print("No AIFS files found; skipping AIFS normalization stats.")
            return

        def factory() -> Any:
            for path in files:
                yield read_aifs_channels(path, channels=config["channels"], domain=domain, aifs_config=config.get("aifs", {}))[0]

    stats = compute_norm_stats_stream(factory, config["channels"], config.get("norm", {}))
    out = Path(paths.get("norm_stats_aifs", ROOT / "outputs" / "norm_stats_aifs.json"))
    save_norm_stats(stats, out)
    print(f"Wrote {out}")


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pretrain.yaml"))
    parser.add_argument("--domain", choices=["era5", "aifs", "all"], default="all")
    parser.add_argument("--smoke-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.domain in {"era5", "all"}:
        _compute_era5(config, smoke=args.smoke_synthetic)
    if args.domain in {"aifs", "all"}:
        _compute_aifs(config, smoke=args.smoke_synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
