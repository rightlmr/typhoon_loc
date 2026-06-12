"""Quiet-period false-alarm baseline for operational threshold selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401

import pandas as pd
import torch

from scripts.ops_threshold_sweep import DEFAULT_THRESHOLDS, _lead_bins_with_all, _parse_thresholds
from scripts.predict import _load_model, _predict_array
from tclocator.common import DomainConfig, iter_files, load_config, resolve_device, set_seed
from tclocator.io_aifs import parse_aifs_filename, read_aifs_channels
from tclocator.labels import read_ibtracs
from tclocator.normalization import apply_norm, load_norm_stats
from tclocator.split import aifs_init_month
from tclocator.tracking import link_tracks


def _lead_bin_name(lead_hour: float) -> str:
    """Return the operational lead-bin name for one lead hour."""

    for lead_bin in _lead_bins_with_all():
        if lead_bin.name != "ALL" and lead_bin.min_hour <= lead_hour < lead_bin.max_hour:
            return lead_bin.name
    return "ALL"


def _quiet_files(config: dict[str, Any], months: set[str]) -> list[Path]:
    """Return AIFS files for quiet months, filtered to configured lead max."""

    files = iter_files(config.get("paths", {}).get("aifs_dir", ""), [".grib2", ".grb2", ".grib", ".pt"])
    lead_max = config.get("finetune", {}).get("lead_max")
    kept: list[Path] = []
    for path in files:
        meta = parse_aifs_filename(path)
        if aifs_init_month(path) not in months:
            continue
        if lead_max is not None and meta.forecast_hour > int(lead_max):
            continue
        kept.append(path)
    return kept


def _assert_truthless_months(config: dict[str, Any], months: set[str]) -> None:
    """Raise if configured quiet months contain in-domain IBTrACS truth."""

    ib_path = Path(config.get("paths", {}).get("ibtracs_csv", ""))
    if not ib_path.exists():
        return
    truth = read_ibtracs(ib_path, config.get("ibtracs", {}).get("col_map", {}))
    if truth.empty:
        return
    truth = truth.copy()
    truth["YEAR_MONTH"] = truth["ISO_TIME"].dt.strftime("%Y-%m")
    counts = truth.loc[truth["YEAR_MONTH"].isin(months)].groupby("YEAR_MONTH").size()
    bad = counts[counts > 0]
    if not bad.empty:
        raise RuntimeError(f"Quiet months contain IBTrACS truth rows: {bad.to_dict()}")


def _predict_quiet(config: dict[str, Any], files: list[Path], *, conf_thresh: float, output_path: Path) -> pd.DataFrame:
    """Run low-threshold inference on quiet AIFS files."""

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    ckpt = Path(config.get("paths", {}).get("finetune_ckpt", ""))
    model = _load_model(config, device, ckpt)
    norm_path = Path(config.get("paths", {}).get("norm_stats_aifs", ""))
    norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None
    domain = DomainConfig.from_mapping(config.get("domain"))
    rows: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        if index % 200 == 0:
            print(f"quiet predict processed={index}/{len(files)}", flush=True)
        field, _ = read_aifs_channels(path, channels=config["channels"], domain=domain, aifs_config=config.get("aifs", {}))
        if norm_stats is not None:
            field = apply_norm(field, norm_stats)
        meta = parse_aifs_filename(path)
        frame = _predict_array(
            model,
            field,
            config,
            device,
            iso_time=meta.valid_time.isoformat(),
            lead_hour=meta.forecast_hour,
            conf_thresh=conf_thresh,
        )
        if not frame.empty:
            frame["YEAR_MONTH"] = meta.init_time.strftime("%Y-%m")
            frame["FIELD_ID"] = path.stem
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ISO_TIME", "LAT", "LON", "CONF", "LEAD_HOUR", "YEAR_MONTH", "FIELD_ID"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _field_counts(files: list[Path]) -> pd.DataFrame:
    """Count quiet fields by month and lead bin."""

    rows: list[dict[str, Any]] = []
    for path in files:
        meta = parse_aifs_filename(path)
        rows.append({"year_month": meta.init_time.strftime("%Y-%m"), "lead_bin": _lead_bin_name(float(meta.forecast_hour))})
        rows.append({"year_month": meta.init_time.strftime("%Y-%m"), "lead_bin": "ALL"})
    return pd.DataFrame(rows).groupby(["year_month", "lead_bin"]).size().rename("n_fields").reset_index()


def _track_summary(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Link tracks and summarize them by month and median lead bin."""

    tracking_cfg = config.get("tracking", {})
    tracks = link_tracks(
        predictions,
        max_step_km=float(tracking_cfg.get("max_step_km", 800.0)),
        min_len=int(tracking_cfg.get("min_len", 4)),
        expected_step_hours=float(tracking_cfg.get("expected_step_hours", 6.0)),
    )
    if tracks.empty:
        return pd.DataFrame(columns=["year_month", "lead_bin", "n_tracks"])
    rows: list[dict[str, Any]] = []
    for _, track in tracks.groupby("TRACK_ID"):
        month = str(track["YEAR_MONTH"].mode().iloc[0]) if "YEAR_MONTH" in track else ""
        median_lead = float(pd.to_numeric(track["LEAD_HOUR"], errors="coerce").median())
        rows.append({"year_month": month, "lead_bin": _lead_bin_name(median_lead), "track_id": int(track["TRACK_ID"].iloc[0])})
        rows.append({"year_month": month, "lead_bin": "ALL", "track_id": int(track["TRACK_ID"].iloc[0])})
    return pd.DataFrame(rows).groupby(["year_month", "lead_bin"])["track_id"].nunique().rename("n_tracks").reset_index()


def _quiet_far_sweep(
    predictions: pd.DataFrame,
    files: list[Path],
    config: dict[str, Any],
    thresholds: list[float],
    *,
    months: list[str],
) -> pd.DataFrame:
    """Calculate quiet-period FAR rows for every threshold."""

    fields = _field_counts(files)
    rows: list[pd.DataFrame] = []
    month_count = len(months)
    for threshold in thresholds:
        pred = predictions.loc[predictions["CONF"] >= float(threshold)].copy() if not predictions.empty else predictions.copy()
        if not pred.empty:
            pred["lead_bin"] = pd.to_numeric(pred["LEAD_HOUR"], errors="coerce").map(lambda value: _lead_bin_name(float(value)))
            pred_all = pred.copy()
            pred_all["lead_bin"] = "ALL"
            pred_for_counts = pd.concat([pred, pred_all], ignore_index=True)
        else:
            pred_for_counts = pred
        point_counts = (
            pred_for_counts.groupby(["YEAR_MONTH", "lead_bin"]).size().rename("n_points").reset_index()
            if not pred_for_counts.empty
            else pd.DataFrame(columns=["YEAR_MONTH", "lead_bin", "n_points"])
        )
        track_counts = _track_summary(pred, config)
        base = fields.copy()
        base = base.merge(point_counts, left_on=["year_month", "lead_bin"], right_on=["YEAR_MONTH", "lead_bin"], how="left")
        base = base.merge(track_counts, on=["year_month", "lead_bin"], how="left")
        base["n_points"] = base["n_points"].fillna(0).astype(int)
        base["n_tracks"] = base["n_tracks"].fillna(0).astype(int)
        base["conf_thresh"] = float(threshold)
        base["raw_points_per_field"] = base["n_points"] / base["n_fields"].clip(lower=1)
        base["tracks_per_month"] = base["n_tracks"].astype(float)
        rows.append(base[["year_month", "conf_thresh", "lead_bin", "raw_points_per_field", "n_tracks", "tracks_per_month"]])

        all_rows = []
        for lead_bin in [bin_.name for bin_ in _lead_bins_with_all()]:
            fields_part = fields.loc[fields["lead_bin"] == lead_bin]
            n_fields = int(fields_part["n_fields"].sum())
            points_part = point_counts.loc[point_counts["lead_bin"] == lead_bin] if not point_counts.empty else point_counts
            tracks_part = track_counts.loc[track_counts["lead_bin"] == lead_bin] if not track_counts.empty else track_counts
            n_points = int(points_part["n_points"].sum()) if not points_part.empty else 0
            n_tracks = int(tracks_part["n_tracks"].sum()) if not tracks_part.empty else 0
            all_rows.append(
                {
                    "year_month": "ALL",
                    "conf_thresh": float(threshold),
                    "lead_bin": lead_bin,
                    "raw_points_per_field": float(n_points / max(n_fields, 1)),
                    "n_tracks": n_tracks,
                    "tracks_per_month": float(n_tracks / max(month_count, 1)),
                }
            )
        rows.append(pd.DataFrame(all_rows))
    return pd.concat(rows, ignore_index=True)


def _far_status(tracks_per_month: float) -> str:
    """Return operational FAR interpretation."""

    if tracks_per_month <= 1.0:
        return "ACCEPTABLE"
    if tracks_per_month <= 3.0:
        return "REVIEW"
    return "RAISE_THRESHOLD"


def _combine_recommendations(far: pd.DataFrame, recommendation_path: Path, output_path: Path) -> pd.DataFrame:
    """Join operating thresholds with quiet-period false-alarm estimates."""

    rec = pd.read_csv(recommendation_path)
    rows: list[dict[str, Any]] = []
    quiet_all = far.loc[far["year_month"] == "ALL"].copy()
    for _, row in rec.iterrows():
        lead_bin = str(row["lead_bin"])
        threshold = float(row["recommended_conf_thresh"])
        part = quiet_all.loc[(quiet_all["lead_bin"] == lead_bin) & (quiet_all["conf_thresh"] == threshold)]
        far_row = part.iloc[0] if not part.empty else None
        tracks_per_month = float(far_row["tracks_per_month"]) if far_row is not None else float("nan")
        rows.append(
            {
                "lead_bin": lead_bin,
                "recommended_conf_thresh": threshold,
                "threshold_status": row["status"],
                "track_precision_2025": row["track_precision"],
                "track_recall_2025": row["track_recall"],
                "quiet_tracks_per_month": tracks_per_month,
                "far_status": _far_status(tracks_per_month) if pd.notna(tracks_per_month) else "UNKNOWN",
            }
        )
    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(out.to_string(index=False))
    return out


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--months", default="2024-04,2025-01,2025-02")
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--conf-thresh", type=float, default=0.05)
    parser.add_argument("--predictions-input", default=None)
    parser.add_argument("--predictions-output", default=str(ROOT / "outputs" / "predictions_quiet_sweep.csv"))
    parser.add_argument("--far-output", default=str(ROOT / "outputs" / "quiet_far_sweep.csv"))
    parser.add_argument("--recommendations", default=str(ROOT / "outputs" / "ops_threshold_recommendation.csv"))
    parser.add_argument("--final-output", default=str(ROOT / "outputs" / "ops_operating_thresholds.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    months = [item.strip() for item in args.months.split(",") if item.strip()]
    _assert_truthless_months(config, set(months))
    files = _quiet_files(config, set(months))
    if not files:
        raise RuntimeError(f"No quiet AIFS files found for months={months}")

    if args.predictions_input:
        predictions = pd.read_csv(args.predictions_input)
    else:
        predictions = _predict_quiet(config, files, conf_thresh=float(args.conf_thresh), output_path=Path(args.predictions_output))
    far = _quiet_far_sweep(predictions, files, config, _parse_thresholds(args.thresholds), months=months)
    far_path = Path(args.far_output)
    far_path.parent.mkdir(parents=True, exist_ok=True)
    far.to_csv(far_path, index=False)
    print(f"Wrote {far_path}")
    _combine_recommendations(far, Path(args.recommendations), Path(args.final_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
