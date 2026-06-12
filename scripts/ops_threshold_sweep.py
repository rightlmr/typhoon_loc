"""Per-lead point and track threshold sweep for operational settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401

import numpy as np
import pandas as pd

from scripts.evaluate import _build_references
from tclocator.common import haversine_km, load_config
from tclocator.metrics import DEFAULT_LEAD_BINS, LeadBin, match_predictions
from tclocator.tracking import link_tracks


DEFAULT_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
ALL_BIN = LeadBin("ALL", 0, 121)


def _lead_bins_with_all() -> tuple[LeadBin, ...]:
    """Return operational lead bins plus an aggregate row."""

    return (*DEFAULT_LEAD_BINS, ALL_BIN)


def _in_bin(df: pd.DataFrame, lead_bin: LeadBin) -> pd.Series:
    """Return a boolean mask for one lead bin."""

    if lead_bin.name == "ALL":
        return pd.Series(True, index=df.index)
    return (df["LEAD_HOUR"] >= lead_bin.min_hour) & (df["LEAD_HOUR"] < lead_bin.max_hour)


def _prediction_count(predictions: pd.DataFrame, lead_bin: LeadBin) -> int:
    """Count predictions in one lead bin."""

    if predictions.empty:
        return 0
    if "LEAD_HOUR" not in predictions.columns or lead_bin.name == "ALL":
        return int(len(predictions))
    return int(_in_bin(predictions, lead_bin).sum())


def _track_lead_bin(track: pd.DataFrame) -> str:
    """Assign one linked track to a lead bin by median lead."""

    if track.empty or "LEAD_HOUR" not in track.columns:
        return "ALL"
    median_lead = float(pd.to_numeric(track["LEAD_HOUR"], errors="coerce").median())
    for lead_bin in DEFAULT_LEAD_BINS:
        if lead_bin.min_hour <= median_lead < lead_bin.max_hour:
            return lead_bin.name
    return "ALL"


def _best_track_sid(
    track: pd.DataFrame,
    references: pd.DataFrame,
    *,
    hit_threshold_km: float,
) -> tuple[str | None, float]:
    """Return the best-matching truth SID and per-track match ratio."""

    if track.empty or references.empty:
        return None, 0.0
    refs = references.copy()
    refs["ISO_TIME"] = pd.to_datetime(refs["ISO_TIME"], utc=True, errors="coerce")
    refs["LEAD_HOUR"] = pd.to_numeric(refs["LEAD_HOUR"], errors="coerce").astype("Int64")
    det = track.copy()
    det["ISO_TIME"] = pd.to_datetime(det["ISO_TIME"], utc=True, errors="coerce")
    det["LEAD_HOUR"] = pd.to_numeric(det["LEAD_HOUR"], errors="coerce").astype("Int64")
    sid_hits: dict[str, int] = {}
    for _, row in det.iterrows():
        same_field = refs.loc[(refs["ISO_TIME"] == row["ISO_TIME"]) & (refs["LEAD_HOUR"] == row["LEAD_HOUR"])]
        if same_field.empty:
            continue
        for _, ref in same_field.iterrows():
            dist = haversine_km(float(row["LAT"]), float(row["LON"]), float(ref["LAT_FIELD"]), float(ref["LON_FIELD"]))
            if float(dist) <= hit_threshold_km:
                sid = str(ref["SID"])
                sid_hits[sid] = sid_hits.get(sid, 0) + 1
    if not sid_hits:
        return None, 0.0
    sid, hits = max(sid_hits.items(), key=lambda item: item[1])
    return sid, float(hits / max(len(det), 1))


def _track_metrics(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    config: dict[str, Any],
    *,
    lead_bin: LeadBin,
    hit_threshold_km: float,
    match_ratio: float,
) -> tuple[int, float, float]:
    """Calculate track count, precision, and recall for one lead bin."""

    tracking_cfg = config.get("tracking", {})
    tracks = link_tracks(
        predictions,
        max_step_km=float(tracking_cfg.get("max_step_km", 800.0)),
        min_len=int(tracking_cfg.get("min_len", 4)),
        expected_step_hours=float(tracking_cfg.get("expected_step_hours", 6.0)),
    )
    if tracks.empty:
        n_truth_sid = int(references.loc[_in_bin(references, lead_bin), "SID"].nunique()) if not references.empty else 0
        return 0, float("nan"), 0.0 if n_truth_sid else float("nan")

    hit_tracks = 0
    hit_sids: set[str] = set()
    n_tracks = 0
    for _, track in tracks.groupby("TRACK_ID"):
        track_bin = _track_lead_bin(track)
        if lead_bin.name != "ALL" and track_bin != lead_bin.name:
            continue
        n_tracks += 1
        sid, ratio = _best_track_sid(track, references, hit_threshold_km=hit_threshold_km)
        if sid is not None and ratio >= match_ratio:
            hit_tracks += 1
            hit_sids.add(sid)
    truth_part = references.loc[_in_bin(references, lead_bin)] if not references.empty else references
    n_truth_sid = int(truth_part["SID"].nunique()) if not truth_part.empty else 0
    precision = float(hit_tracks / n_tracks) if n_tracks > 0 else float("nan")
    recall = float(len(hit_sids) / n_truth_sid) if n_truth_sid > 0 else float("nan")
    return n_tracks, precision, recall


def sweep_dataset(
    *,
    dataset: str,
    predictions_path: Path,
    config_path: Path,
    split: str,
    thresholds: Iterable[float],
    hit_threshold_km: float,
    match_ratio: float,
) -> pd.DataFrame:
    """Run point-level and track-level threshold sweep for one dataset."""

    config = load_config(config_path)
    predictions = pd.read_csv(predictions_path)
    references = _build_references(config, split)
    if references.empty:
        raise RuntimeError(f"No references found for {dataset} using {config_path} split={split}")

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        pred = predictions.loc[predictions["CONF"] >= float(threshold)].copy()
        matched = match_predictions(pred, references, hit_threshold_km=hit_threshold_km)
        for lead_bin in _lead_bins_with_all():
            ref_part = references.loc[_in_bin(references, lead_bin)]
            matched_part = matched.loc[_in_bin(matched, lead_bin)] if not matched.empty else matched
            n_truth = int(len(ref_part))
            n_pred = _prediction_count(pred, lead_bin)
            hits = int(matched_part["hit"].sum()) if not matched_part.empty else 0
            n_tracks, track_precision, track_recall = _track_metrics(
                pred,
                references,
                config,
                lead_bin=lead_bin,
                hit_threshold_km=hit_threshold_km,
                match_ratio=match_ratio,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "conf_thresh": float(threshold),
                    "lead_bin": lead_bin.name,
                    "n_truth": n_truth,
                    "n_predictions": n_pred,
                    "point_precision": float(hits / n_pred) if n_pred > 0 else float("nan"),
                    "point_recall": float(hits / n_truth) if n_truth > 0 else float("nan"),
                    "n_tracks": n_tracks,
                    "track_precision": track_precision,
                    "track_recall": track_recall,
                }
            )
    return pd.DataFrame(rows)


def _recommend_thresholds(sweep: pd.DataFrame, *, primary_dataset: str, target_precision: float) -> pd.DataFrame:
    """Pick the lowest threshold that reaches target track precision per lead bin."""

    rows: list[dict[str, Any]] = []
    primary = sweep.loc[sweep["dataset"] == primary_dataset].copy()
    for lead_bin in [bin_.name for bin_ in _lead_bins_with_all()]:
        part = primary.loc[primary["lead_bin"] == lead_bin].sort_values("conf_thresh")
        eligible = part.loc[(part["n_tracks"] > 0) & (part["track_precision"] >= target_precision)]
        if not eligible.empty:
            chosen = eligible.iloc[0]
            status = "OK"
        else:
            ranked = part.copy()
            ranked["_rank_precision"] = ranked["track_precision"].fillna(-1.0)
            chosen = ranked.sort_values(["_rank_precision", "track_recall", "conf_thresh"], ascending=[False, False, True]).iloc[0]
            status = "UNREACHABLE"
        rows.append(
            {
                "lead_bin": lead_bin,
                "recommended_conf_thresh": float(chosen["conf_thresh"]),
                "status": status,
                "target_track_precision": float(target_precision),
                "track_precision": float(chosen["track_precision"]) if pd.notna(chosen["track_precision"]) else float("nan"),
                "track_recall": float(chosen["track_recall"]) if pd.notna(chosen["track_recall"]) else float("nan"),
                "point_precision": float(chosen["point_precision"]) if pd.notna(chosen["point_precision"]) else float("nan"),
                "point_recall": float(chosen["point_recall"]) if pd.notna(chosen["point_recall"]) else float("nan"),
                "n_tracks": int(chosen["n_tracks"]),
            }
        )
    return pd.DataFrame(rows)


def _parse_thresholds(raw: str) -> list[float]:
    """Parse comma-separated threshold values."""

    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-2025", default=str(ROOT / "outputs" / "predictions_2025_sweep.csv"))
    parser.add_argument("--config-2025", default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--split-2025", default="all", choices=["all", "train", "val"])
    parser.add_argument("--predictions-2024val", default=str(ROOT / "outputs" / "predictions_2024val_sweep.csv"))
    parser.add_argument("--config-2024val", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--split-2024val", default="val", choices=["all", "train", "val"])
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--target-precision", type=float, default=0.7)
    parser.add_argument("--hit-threshold-km", type=float, default=60.0)
    parser.add_argument("--track-match-ratio", type=float, default=0.5)
    parser.add_argument("--output-2025", default=str(ROOT / "outputs" / "ops_sweep_2025.csv"))
    parser.add_argument("--output-2024val", default=str(ROOT / "outputs" / "ops_sweep_2024val.csv"))
    parser.add_argument("--recommendation-output", default=str(ROOT / "outputs" / "ops_threshold_recommendation.csv"))
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    jobs = [
        ("2025", Path(args.predictions_2025), Path(args.config_2025), args.split_2025, Path(args.output_2025)),
        ("2024val", Path(args.predictions_2024val), Path(args.config_2024val), args.split_2024val, Path(args.output_2024val)),
    ]
    frames: list[pd.DataFrame] = []
    for dataset, predictions_path, config_path, split, output_path in jobs:
        frame = sweep_dataset(
            dataset=dataset,
            predictions_path=predictions_path,
            config_path=config_path,
            split=split,
            thresholds=thresholds,
            hit_threshold_km=float(args.hit_threshold_km),
            match_ratio=float(args.track_match_ratio),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        frames.append(frame)
        print(f"Wrote {output_path}")

    combined = pd.concat(frames, ignore_index=True)
    recommendation = _recommend_thresholds(combined, primary_dataset="2025", target_precision=float(args.target_precision))
    rec_path = Path(args.recommendation_output)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation.to_csv(rec_path, index=False)
    print(f"Wrote {rec_path}")
    print(recommendation.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
