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
from tclocator.labels import find_field_min_center, find_hybrid_center, read_ibtracs, records_at_time
from tclocator.metrics import DEFAULT_LEAD_BINS, match_predictions, precision_recall_curve, summarize_by_lead
from tclocator.split import select_aifs_files


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


def _build_references(config: dict[str, Any], split: str) -> pd.DataFrame:
    """Build AIFS references with true and field centers."""

    paths = config.get("paths", {})
    files = iter_files(paths.get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
    files = select_aifs_files(config, files, split)
    ib_path = Path(paths.get("ibtracs_csv", ""))
    if not files or not ib_path.exists():
        return pd.DataFrame()
    domain = DomainConfig.from_mapping(config.get("domain"))
    label_cfg = config.get("labels", {})
    label_mode = str(label_cfg.get("mode", "in_field"))
    center_criterion = str(label_cfg.get("center_criterion", "msl"))
    records = read_ibtracs(ib_path, config.get("ibtracs", {}).get("col_map", {}))
    rows: list[dict[str, Any]] = []
    for path in files:
        meta = parse_aifs_filename(path)
        at_time = records_at_time(records, pd.Timestamp(meta.valid_time))
        if at_time.empty:
            continue
        field = None
        if label_mode == "in_field":
            channels = ["msl", "vo_850"] if center_criterion == "msl_vo_hybrid" else ["msl"]
            field, _ = read_aifs_channels(path, channels=channels, domain=domain, aifs_config=config.get("aifs", {}))
        for _, record in at_time.iterrows():
            lat_true = float(record["LAT"])
            lon_true = float(record["LON"])
            if label_mode == "ibtracs":
                lat_field, lon_field = lat_true, lon_true
            elif label_mode == "in_field":
                if field is None:
                    raise RuntimeError("MSL field was not loaded for in_field evaluation")
                if center_criterion == "msl_vo_hybrid":
                    lat_field, lon_field, center_source = find_hybrid_center(
                        field[0],
                        field[1],
                        lat_true,
                        lon_true,
                        domain,
                        float(label_cfg.get("search_radius_km", 300.0)),
                        field_center_smooth_px=int(label_cfg.get("field_center_smooth_px", 0)),
                        vo_smooth_px=int(label_cfg.get("vo_smooth_px", 1)),
                    )
                else:
                    lat_field, lon_field = find_field_min_center(
                        field[0],
                        lat_true,
                        lon_true,
                        domain,
                        float(label_cfg.get("search_radius_km", 300.0)),
                        smooth_px=int(label_cfg.get("field_center_smooth_px", 0)),
                    )
                    center_source = "msl"
            else:
                raise ValueError("labels.mode must be 'ibtracs' or 'in_field'")
            rows.append(
                {
                    "ISO_TIME": meta.valid_time.isoformat(),
                    "SID": record["SID"],
                    "LEAD_HOUR": meta.forecast_hour,
                    "YEAR_MONTH": meta.init_time.strftime("%Y-%m"),
                    "LAT_TRUE": lat_true,
                    "LON_TRUE": lon_true,
                    "LAT_FIELD": lat_field,
                    "LON_FIELD": lon_field,
                    "CENTER_SOURCE": center_source if label_mode == "in_field" else "ibtracs",
                }
            )
    return pd.DataFrame(rows)


def _with_split_suffix(path: Path, split: str) -> Path:
    """Return an output path with a split suffix when requested."""

    if split == "all":
        return path
    return path.with_name(f"{path.stem}_{split}{path.suffix}")


def _attach_reference_month(matched: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Attach AIFS initialization month to matched rows."""

    if matched.empty or "YEAR_MONTH" not in references.columns:
        return matched
    ref_months = references[["ISO_TIME", "SID", "LEAD_HOUR", "YEAR_MONTH"]].copy()
    ref_months["ISO_TIME"] = pd.to_datetime(ref_months["ISO_TIME"], utc=True, errors="coerce")
    ref_months["LEAD_HOUR"] = pd.to_numeric(ref_months["LEAD_HOUR"], errors="coerce").astype("Int64")
    out = matched.copy()
    out["ISO_TIME"] = pd.to_datetime(out["ISO_TIME"], utc=True, errors="coerce")
    out["LEAD_HOUR"] = pd.to_numeric(out["LEAD_HOUR"], errors="coerce").astype("Int64")
    return out.merge(ref_months, on=["ISO_TIME", "SID", "LEAD_HOUR"], how="left")


def _summarize_by_month(matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched metrics by AIFS initialization month."""

    rows: list[dict[str, Any]] = []
    if matched.empty or "YEAR_MONTH" not in matched.columns:
        return pd.DataFrame(rows)
    for year_month, part in matched.groupby("YEAR_MONTH"):
        rows.append(
            {
                "year_month": str(year_month),
                "n_ref": int(len(part)),
                "recall": float(part["hit"].mean()),
                "loc_error_median_km": float(part["loc_error_km"].median(skipna=True)),
                "track_bias_median_km": float(part["track_bias_km"].median(skipna=True)),
                "end2end_median_km": float(part["end2end_km"].median(skipna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("year_month").reset_index(drop=True)


def _summarize_by_month_lead(matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched metrics by AIFS initialization month and lead bin."""

    rows: list[dict[str, Any]] = []
    if matched.empty or "YEAR_MONTH" not in matched.columns:
        return pd.DataFrame(rows)
    for year_month, month_part in matched.groupby("YEAR_MONTH"):
        for lead_bin in DEFAULT_LEAD_BINS:
            part = month_part.loc[(month_part["LEAD_HOUR"] >= lead_bin.min_hour) & (month_part["LEAD_HOUR"] < lead_bin.max_hour)]
            if part.empty:
                continue
            rows.append(
                {
                    "year_month": str(year_month),
                    "lead_bin": lead_bin.name,
                    "n_ref": int(len(part)),
                    "recall": float(part["hit"].mean()),
                    "loc_error_median_km": float(part["loc_error_km"].median(skipna=True)),
                    "track_bias_median_km": float(part["track_bias_km"].median(skipna=True)),
                    "end2end_median_km": float(part["end2end_km"].median(skipna=True)),
                }
            )
    return pd.DataFrame(rows).sort_values(["year_month", "lead_bin"]).reset_index(drop=True)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
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
        references = _build_references(config, args.split)
        if references.empty:
            print("No AIFS/IBTrACS references found; evaluation skipped.")
            return 0

    matched = match_predictions(predictions, references)
    matched = _attach_reference_month(matched, references)
    summary = summarize_by_lead(matched)
    by_month = _summarize_by_month(matched)
    by_month_lead = _summarize_by_month_lead(matched)
    pr = precision_recall_curve(predictions, references, thresholds=[0.1, 0.2, 0.3, 0.5, 0.7])
    out_dir = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_path = _with_split_suffix(out_dir / "matched_metrics.csv", args.split)
    summary_path = _with_split_suffix(out_dir / "metrics_by_lead.csv", args.split)
    month_path = _with_split_suffix(out_dir / "metrics_by_month.csv", args.split)
    month_lead_path = _with_split_suffix(out_dir / "metrics_by_month_lead.csv", args.split)
    pr_path = _with_split_suffix(out_dir / "precision_recall.csv", args.split)
    matched.to_csv(matched_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_month.to_csv(month_path, index=False)
    by_month_lead.to_csv(month_lead_path, index=False)
    pr.to_csv(pr_path, index=False)
    print(f"Wrote {matched_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {month_path}")
    print(f"Wrote {month_lead_path}")
    print(f"Wrote {pr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
