"""Evaluate decoded predictions with the three-way TC error decomposition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
import pandas as pd

from tclocator.common import DomainConfig, iter_files, load_config
from tclocator.labels import find_field_min_center, read_ibtracs, records_at_time
from tclocator.metrics import match_predictions, precision_recall_curve, summarize_by_lead


def _synthetic_refs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return synthetic predictions and references for smoke testing."""

    refs = pd.DataFrame(
        [
            {
                "ISO_TIME": "2000-01-01T00:00:00Z",
                "SID": "SYN001",
                "LEAD_HOUR": 0,
                "LAT_TRUE": 12.0,
                "LON_TRUE": 130.0,
                "LAT_FIELD": 12.1,
                "LON_FIELD": 130.1,
            }
        ]
    )
    preds = pd.DataFrame([{"ISO_TIME": "2000-01-01T00:00:00Z", "LAT": 12.12, "LON": 130.08, "CONF": 0.9}])
    return preds, refs


def _build_references(config: dict[str, Any]) -> pd.DataFrame:
    """Build AIFS references with true and field centers."""

    paths = config.get("paths", {})
    files = iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib"])
    ib_path = Path(paths.get("ibtracs_csv", ""))
    if not files or not ib_path.exists():
        return pd.DataFrame()
    domain = DomainConfig.from_mapping(config.get("domain"))
    records = read_ibtracs(ib_path, config.get("ibtracs", {}).get("col_map", {}))
    rows: list[dict[str, Any]] = []
    for path in files:
        meta = parse_aifs_filename(path)
        at_time = records_at_time(records, pd.Timestamp(meta.valid_time))
        if at_time.empty:
            continue
        field, _ = read_aifs_channels(path, channels=["msl"], domain=domain, aifs_config=config.get("aifs", {}))
        for _, record in at_time.iterrows():
            lat_field, lon_field = find_field_min_center(
                field[0],
                float(record["LAT"]),
                float(record["LON"]),
                domain,
                float(config.get("labels", {}).get("search_radius_km", 300.0)),
            )
            rows.append(
                {
                    "ISO_TIME": meta.valid_time.isoformat(),
                    "SID": record["SID"],
                    "LEAD_HOUR": meta.forecast_hour,
                    "LAT_TRUE": float(record["LAT"]),
                    "LON_TRUE": float(record["LON"]),
                    "LAT_FIELD": lat_field,
                    "LON_FIELD": lon_field,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--smoke-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke_synthetic:
        predictions, references = _synthetic_refs()
    else:
        pred_path = Path(args.predictions or config.get("paths", {}).get("predictions_csv", ""))
        if not pred_path.exists():
            print(f"Predictions file not found: {pred_path}")
            return 1
        predictions = pd.read_csv(pred_path)
        references = _build_references(config)
        if references.empty:
            print("No AIFS/IBTrACS references found; evaluation skipped.")
            return 0

    matched = match_predictions(predictions, references)
    summary = summarize_by_lead(matched)
    pr = precision_recall_curve(predictions, references, thresholds=[0.1, 0.2, 0.3, 0.5, 0.7])
    out_dir = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_path = out_dir / "matched_metrics.csv"
    summary_path = out_dir / "metrics_by_lead.csv"
    pr_path = out_dir / "precision_recall.csv"
    matched.to_csv(matched_path, index=False)
    summary.to_csv(summary_path, index=False)
    pr.to_csv(pr_path, index=False)
    print(f"Wrote {matched_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {pr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
