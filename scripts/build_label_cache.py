"""Build compressed label caches for ERA5 or AIFS fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator.common import DomainConfig, PHASE0_REQUIRED_MESSAGE, haversine_km, iter_files, load_config
from tclocator.dataset import SyntheticTCDataset, label_cache_path, parse_era5_valid_time_from_name
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
from tclocator.io_era5 import read_era5_channels
from tclocator.labels import generate_labels, read_ibtracs, records_at_time, save_label_npz
from tclocator.metrics import DEFAULT_LEAD_BINS
from tclocator.split import filter_aifs_files_by_usable_months
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


def _label_channels(label_config: dict[str, Any]) -> list[str]:
    """Return the minimal field channels needed for label generation."""

    if str(label_config.get("center_criterion", "msl")) == "msl_vo_hybrid":
        return ["msl", "vo_850"]
    return ["msl"]


def _append_center_stats(rows: list[dict[str, Any]], label: dict[str, Any], lead_hour: int | None) -> None:
    """Append per-target center-source diagnostics from a generated label."""

    sources = label.get("center_source")
    if sources is None or len(sources) == 0:
        return
    true_lat = label.get("true_lat")
    true_lon = label.get("true_lon")
    center_lat = label.get("center_lat")
    center_lon = label.get("center_lon")
    if true_lat is None or true_lon is None or center_lat is None or center_lon is None:
        return
    for idx, source in enumerate(sources):
        dist = haversine_km(float(true_lat[idx]), float(true_lon[idx]), float(center_lat[idx]), float(center_lon[idx]))
        rows.append(
            {
                "lead_hour": int(lead_hour or 0),
                "center_source": str(source),
                "center_to_truth_km": float(dist),
            }
        )


def _write_center_stats(config: dict[str, Any], rows: list[dict[str, Any]], domain_name: str) -> None:
    """Write lead-binned center-source diagnostics."""

    out_dir = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"label_center_sources_{domain_name}.csv"
    summary_path = out_dir / f"label_center_source_by_lead_{domain_name}.csv"
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    if not raw.empty:
        for lead_bin in DEFAULT_LEAD_BINS:
            part = raw.loc[(raw["lead_hour"] >= lead_bin.min_hour) & (raw["lead_hour"] < lead_bin.max_hour)]
            if part.empty:
                continue
            vo_part = part.loc[part["center_source"] == "vo"]
            summary_rows.append(
                {
                    "lead_bin": lead_bin.name,
                    "n": int(len(part)),
                    "vo_fraction": float((part["center_source"] == "vo").mean()),
                    "vo_center_to_truth_median_km": float(vo_part["center_to_truth_km"].median(skipna=True)) if not vo_part.empty else float("nan"),
                    "all_center_to_truth_median_km": float(part["center_to_truth_km"].median(skipna=True)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


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
    center_stats: list[dict[str, Any]] = []
    label_channels = _label_channels(config.get("labels", {}))
    for path in files:
        valid_time = parse_era5_valid_time_from_name(path)
        if valid_time is None:
            print(f"Skip {path}: cannot parse valid time from filename")
            continue
        field, _ = read_era5_channels(path, channels=label_channels, domain=domain, era5_config=config.get("era5", {}))
        label = generate_labels(
            msl=field[label_channels.index("msl")],
            vo=field[label_channels.index("vo_850")] if "vo_850" in label_channels else None,
            records=records_at_time(records, valid_time),
            domain=domain,
            label_config=config.get("labels", {}),
        )
        _append_center_stats(center_stats, label, None)
        out = label_cache_path(config, "era5", path)
        save_label_npz(label, out)
        print(f"Wrote {out}")
    _write_center_stats(config, center_stats, "era5")


def _build_aifs(config: dict[str, Any]) -> None:
    """Build AIFS label caches."""

    paths = config.get("paths", {})
    files = filter_aifs_files_by_usable_months(
        config,
        iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"]),
    )
    lead_max = config.get("finetune", {}).get("lead_max")
    if lead_max is not None:
        max_hour = int(lead_max)
        files = [path for path in files if parse_aifs_filename(path).forecast_hour <= max_hour]
    ibtracs_path = Path(paths.get("ibtracs_csv", ""))
    if not files or not ibtracs_path.exists():
        print("No AIFS/IBTrACS data found; skipping AIFS label cache.")
        return
    domain = DomainConfig.from_mapping(config.get("domain"))
    records = read_ibtracs(ibtracs_path, config.get("ibtracs", {}).get("col_map", {}))
    center_stats: list[dict[str, Any]] = []
    label_channels = _label_channels(config.get("labels", {}))
    for path in files:
        meta = parse_aifs_filename(path)
        field, _ = read_aifs_channels(path, channels=label_channels, domain=domain, aifs_config=config.get("aifs", {}))
        label = generate_labels(
            msl=field[0],
            vo=field[label_channels.index("vo_850")] if "vo_850" in label_channels else None,
            records=records_at_time(records, pd.Timestamp(meta.valid_time)),
            domain=domain,
            label_config=config.get("labels", {}),
        )
        _append_center_stats(center_stats, label, meta.forecast_hour)
        out = label_cache_path(config, "aifs", path)
        save_label_npz(label, out)
        print(f"Wrote {out}")
    _write_center_stats(config, center_stats, "aifs")


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
