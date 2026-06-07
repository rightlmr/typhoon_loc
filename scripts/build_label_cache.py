"""Build compressed label caches for ERA5 or AIFS fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator.common import DomainConfig, PHASE0_REQUIRED_MESSAGE, iter_files, load_config
from tclocator.dataset import SyntheticTCDataset, label_cache_path, parse_era5_valid_time_from_name
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
from tclocator.io_era5 import read_era5_channels
from tclocator.labels import generate_labels, read_ibtracs, records_at_time, save_label_npz
import pandas as pd


def _assert_labels_ready(config: dict[str, Any]) -> bool:
    """Check labels.mode and print the Phase 0 message when missing."""

    if config.get("labels", {}).get("mode") is None:
        print(PHASE0_REQUIRED_MESSAGE)
        return False
    return True


def _smoke(config: dict[str, Any]) -> None:
    """Build synthetic label caches."""

    out_root = Path(config.get("paths", {}).get("label_cache_dir", ROOT / "data" / "label_cache")) / "smoke"
    dataset = SyntheticTCDataset(length=3, channels=config["channels"], seed=int(config.get("seed", 42)))
    for idx in range(len(dataset)):
        sample = dataset[idx]
        path = out_root / f"synthetic_{idx:03d}.npz"
        save_label_npz(
            {"heatmap": sample["heatmap"].numpy()[0], "offset": sample["offset"].numpy(), "mask": sample["mask"].numpy()},
            path,
        )
        print(f"Wrote {path}")


def _build_era5(config: dict[str, Any]) -> None:
    """Build ERA5 label caches."""

    paths = config.get("paths", {})
    files = iter_files(paths.get("era5_dir", ""), [".nc"])
    ibtracs_path = Path(paths.get("ibtracs_csv", ""))
    if not files or not ibtracs_path.exists():
        print("No ERA5/IBTrACS data found; skipping ERA5 label cache.")
        return
    domain = DomainConfig.from_mapping(config.get("domain"))
    records = read_ibtracs(ibtracs_path, config.get("ibtracs", {}).get("col_map", {}))
    for path in files:
        valid_time = parse_era5_valid_time_from_name(path)
        if valid_time is None:
            print(f"Skip {path}: cannot parse valid time from filename")
            continue
        field, _ = read_era5_channels(path, channels=config["channels"], domain=domain, era5_config=config.get("era5", {}))
        if "msl" not in config["channels"]:
            raise ValueError("Label cache generation requires msl channel")
        label = generate_labels(
            msl=field[config["channels"].index("msl")],
            records=records_at_time(records, valid_time),
            domain=domain,
            label_config=config.get("labels", {}),
        )
        out = label_cache_path(config, "era5", path)
        save_label_npz(label, out)
        print(f"Wrote {out}")


def _build_aifs(config: dict[str, Any]) -> None:
    """Build AIFS label caches."""

    paths = config.get("paths", {})
    files = iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
    ibtracs_path = Path(paths.get("ibtracs_csv", ""))
    if not files or not ibtracs_path.exists():
        print("No AIFS/IBTrACS data found; skipping AIFS label cache.")
        return
    domain = DomainConfig.from_mapping(config.get("domain"))
    records = read_ibtracs(ibtracs_path, config.get("ibtracs", {}).get("col_map", {}))
    for path in files:
        meta = parse_aifs_filename(path)
        field, _ = read_aifs_channels(path, channels=config["channels"], domain=domain, aifs_config=config.get("aifs", {}))
        label = generate_labels(
            msl=field[config["channels"].index("msl")],
            records=records_at_time(records, pd.Timestamp(meta.valid_time)),
            domain=domain,
            label_config=config.get("labels", {}),
        )
        out = label_cache_path(config, "aifs", path)
        save_label_npz(label, out)
        print(f"Wrote {out}")


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pretrain.yaml"))
    parser.add_argument("--domain", choices=["era5", "aifs", "all"], default="all")
    parser.add_argument("--smoke-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke_synthetic:
        _smoke(config)
        return 0
    if not _assert_labels_ready(config):
        return 1
    if args.domain in {"era5", "all"}:
        _build_era5(config)
    if args.domain in {"aifs", "all"}:
        _build_aifs(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
