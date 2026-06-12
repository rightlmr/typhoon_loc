"""Composite per-lead threshold policy sweep for frozen deployment config."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from scripts.ops_threshold_sweep import _best_track_sid
from tclocator.common import haversine_km, load_config
from tclocator.metrics import DEFAULT_LEAD_BINS, LeadBin, match_predictions
from tclocator.tracking import link_tracks


T0_24 = (0.3, 0.4, 0.5, 0.6)
T24_48 = (0.4, 0.5)
T48_96 = (0.6, 0.7, 0.8)
T96_120 = (0.6, 0.8, "EXCLUDE")
OPS1_BASELINE = (0.6, 0.5, 0.8, 0.8)
POLICY_LEAD_BINS = (*DEFAULT_LEAD_BINS, LeadBin("120", 120, 121))


@dataclass(frozen=True)
class Policy:
    """One composite operating threshold policy."""

    policy_id: str
    t_0_24: float
    t_24_48: float
    t_48_96: float
    t_96_120: float | str

    @property
    def threshold_score(self) -> float:
        """Return a tie-break score where lower means more permissive."""

        t96 = 1.0 if str(self.t_96_120).upper() == "EXCLUDE" else float(self.t_96_120)
        return float(self.t_0_24 + self.t_24_48 + self.t_48_96 + t96)


def _policies() -> list[Policy]:
    """Return the 72-policy grid plus an explicit Ops-1 baseline row."""

    rows: list[Policy] = []
    index = 0
    for t0 in T0_24:
        for t24 in T24_48:
            for t48 in T48_96:
                for t96 in T96_120:
                    rows.append(Policy(f"grid_{index:03d}", float(t0), float(t24), float(t48), t96))
                    index += 1
    rows.append(Policy("ops1_baseline", OPS1_BASELINE[0], OPS1_BASELINE[1], OPS1_BASELINE[2], OPS1_BASELINE[3]))
    return rows


def _threshold_for_lead(policy: Policy, lead_hour: float) -> float | None:
    """Return threshold for one forecast lead, or ``None`` when excluded."""

    if 0 <= lead_hour < 24:
        return policy.t_0_24
    if 24 <= lead_hour < 48:
        return policy.t_24_48
    if 48 <= lead_hour < 96:
        return policy.t_48_96
    if 96 <= lead_hour <= 120:
        return None if str(policy.t_96_120).upper() == "EXCLUDE" else float(policy.t_96_120)
    return None


def _apply_policy(predictions: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """Filter detections by their own lead-specific threshold."""

    if predictions.empty:
        return predictions.copy()
    df = predictions.copy()
    leads = pd.to_numeric(df["LEAD_HOUR"], errors="coerce")
    keep = np.zeros(len(df), dtype=bool)
    for idx, lead in enumerate(leads):
        if pd.isna(lead):
            continue
        threshold = _threshold_for_lead(policy, float(lead))
        if threshold is not None and float(df.iloc[idx]["CONF"]) >= threshold:
            keep[idx] = True
    return df.loc[keep].reset_index(drop=True)


def _lead_mask(df: pd.DataFrame, lead_bin: LeadBin) -> pd.Series:
    """Return a lead-bin mask for references or predictions."""

    return (df["LEAD_HOUR"] >= lead_bin.min_hour) & (df["LEAD_HOUR"] < lead_bin.max_hour)


def _linked_tracks(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Run the existing tracker with unchanged configured parameters."""

    tracking_cfg = config.get("tracking", {})
    return link_tracks(
        predictions,
        max_step_km=float(tracking_cfg.get("max_step_km", 800.0)),
        min_len=int(tracking_cfg.get("min_len", 4)),
        expected_step_hours=float(tracking_cfg.get("expected_step_hours", 6.0)),
    )


def _track_hits(
    tracks: pd.DataFrame,
    references: pd.DataFrame,
    *,
    hit_threshold_km: float,
    track_match_ratio: float,
) -> tuple[int, set[str], dict[int, str]]:
    """Return hit-track count, hit SIDs, and track-to-SID assignments."""

    hit_tracks = 0
    hit_sids: set[str] = set()
    assignments: dict[int, str] = {}
    if tracks.empty:
        return hit_tracks, hit_sids, assignments
    for track_id, track in tracks.groupby("TRACK_ID"):
        sid, ratio = _best_track_sid(track, references, hit_threshold_km=hit_threshold_km)
        if sid is not None and ratio >= track_match_ratio:
            hit_tracks += 1
            hit_sids.add(sid)
            assignments[int(track_id)] = sid
    return hit_tracks, hit_sids, assignments


def _coverage_by_lead(
    tracks: pd.DataFrame,
    references: pd.DataFrame,
    hit_sids: set[str],
    assignments: dict[int, str],
    *,
    hit_threshold_km: float,
) -> dict[str, float]:
    """Calculate coverage@lead over storms that have at least one hit track."""

    coverage: dict[str, float] = {}
    if not hit_sids:
        return {f"coverage_{lead_bin.name.replace('-', '_')}": float("nan") for lead_bin in DEFAULT_LEAD_BINS}
    refs = references.copy()
    refs["ISO_TIME"] = pd.to_datetime(refs["ISO_TIME"], utc=True, errors="coerce")
    refs["LEAD_HOUR"] = pd.to_numeric(refs["LEAD_HOUR"], errors="coerce").astype("Int64")
    hit_tracks = tracks.loc[tracks["TRACK_ID"].isin(assignments)].copy() if not tracks.empty else tracks.copy()
    if not hit_tracks.empty:
        hit_tracks["ISO_TIME"] = pd.to_datetime(hit_tracks["ISO_TIME"], utc=True, errors="coerce")
        hit_tracks["LEAD_HOUR"] = pd.to_numeric(hit_tracks["LEAD_HOUR"], errors="coerce").astype("Int64")

    for lead_bin in DEFAULT_LEAD_BINS:
        key = f"coverage_{lead_bin.name.replace('-', '_')}"
        ref_part = refs.loc[refs["SID"].astype(str).isin(hit_sids) & _lead_mask(refs, lead_bin)]
        if ref_part.empty:
            coverage[key] = float("nan")
            continue
        covered = 0
        for _, ref in ref_part.iterrows():
            matched = False
            candidates = hit_tracks.loc[(hit_tracks["ISO_TIME"] == ref["ISO_TIME"]) & (hit_tracks["LEAD_HOUR"] == ref["LEAD_HOUR"])]
            for _, point in candidates.iterrows():
                assigned_sid = assignments.get(int(point["TRACK_ID"]))
                if assigned_sid != str(ref["SID"]):
                    continue
                dist = haversine_km(float(point["LAT"]), float(point["LON"]), float(ref["LAT_FIELD"]), float(ref["LON_FIELD"]))
                if float(dist) <= hit_threshold_km:
                    matched = True
                    break
            if matched:
                covered += 1
        coverage[key] = float(covered / len(ref_part))
    return coverage


def _point_metrics(predictions: pd.DataFrame, references: pd.DataFrame, *, hit_threshold_km: float) -> tuple[float, float]:
    """Return point precision and recall for policy-level context."""

    matched = match_predictions(predictions, references, hit_threshold_km=hit_threshold_km)
    hits = int(matched["hit"].sum()) if not matched.empty else 0
    point_precision = float(hits / len(predictions)) if len(predictions) > 0 else float("nan")
    point_recall = float(hits / len(references)) if len(references) > 0 else float("nan")
    return point_precision, point_recall


def _policy_fields(policy: Policy) -> dict[str, Any]:
    """Return CSV fields that identify a policy."""

    return {
        "policy_id": policy.policy_id,
        "t_0_24": policy.t_0_24,
        "t_24_48": policy.t_24_48,
        "t_48_96": policy.t_48_96,
        "t_96_120": policy.t_96_120,
        "threshold_score": policy.threshold_score,
    }


def evaluate_policy_dataset(
    *,
    dataset: str,
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    config: dict[str, Any],
    policies: Iterable[Policy],
    hit_threshold_km: float,
    track_match_ratio: float,
) -> pd.DataFrame:
    """Evaluate composite policies on a truth-labelled dataset."""

    rows: list[dict[str, Any]] = []
    n_truth_sids = int(references["SID"].nunique()) if not references.empty else 0
    for policy in policies:
        filtered = _apply_policy(predictions, policy)
        tracks = _linked_tracks(filtered, config)
        hit_tracks, hit_sids, assignments = _track_hits(
            tracks,
            references,
            hit_threshold_km=hit_threshold_km,
            track_match_ratio=track_match_ratio,
        )
        point_precision, point_recall = _point_metrics(filtered, references, hit_threshold_km=hit_threshold_km)
        n_tracks = int(tracks["TRACK_ID"].nunique()) if not tracks.empty else 0
        row = {
            "dataset": dataset,
            **_policy_fields(policy),
            "n_points": int(len(filtered)),
            "n_tracks": n_tracks,
            "hit_tracks": int(hit_tracks),
            "n_truth_sids": n_truth_sids,
            "storm_recall": float(len(hit_sids) / n_truth_sids) if n_truth_sids > 0 else float("nan"),
            "track_precision": float(hit_tracks / n_tracks) if n_tracks > 0 else float("nan"),
            "point_precision": point_precision,
            "point_recall": point_recall,
        }
        row.update(_coverage_by_lead(tracks, references, hit_sids, assignments, hit_threshold_km=hit_threshold_km))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_quiet_policies(
    *,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    policies: Iterable[Policy],
) -> pd.DataFrame:
    """Evaluate quiet-period FAR for composite policies."""

    months = sorted(predictions["YEAR_MONTH"].dropna().astype(str).unique().tolist()) if "YEAR_MONTH" in predictions.columns else []
    month_count = max(len(months), 1)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        filtered = _apply_policy(predictions, policy)
        tracks = _linked_tracks(filtered, config)
        n_tracks = int(tracks["TRACK_ID"].nunique()) if not tracks.empty else 0
        per_month = tracks.groupby("YEAR_MONTH")["TRACK_ID"].nunique().to_dict() if not tracks.empty and "YEAR_MONTH" in tracks else {}
        rows.append(
            {
                **_policy_fields(policy),
                "n_points": int(len(filtered)),
                "n_tracks": n_tracks,
                "quiet_months": ",".join(months),
                "quiet_tracks_per_month": float(n_tracks / month_count),
                "quiet_tracks_max_month": float(max(per_month.values(), default=0)),
            }
        )
    return pd.DataFrame(rows)


def _load_truth_dataset(config_path: Path, predictions_path: Path, split: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load predictions, references, and config for one labelled dataset."""

    config = load_config(config_path)
    predictions = pd.read_csv(predictions_path)
    references = _build_references(config, split)
    if references.empty:
        raise RuntimeError(f"No references found for {dataset} using {config_path} split={split}")
    return predictions, references, config


def _select_winner(sweep_2025: pd.DataFrame, quiet: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, str]:
    """Select the deployment policy using the Ops-2 rule or a labelled fallback."""

    merged = sweep_2025.merge(
        quiet[["policy_id", "quiet_tracks_per_month"]],
        on="policy_id",
        how="left",
        validate="one_to_one",
    )
    eligible = merged.loc[(merged["track_precision"] >= 0.7) & (merged["quiet_tracks_per_month"] <= 1.0)].copy()
    if eligible.empty:
        precision_ok = merged.loc[merged["track_precision"] >= 0.7].copy()
        ranked = precision_ok if not precision_ok.empty else merged.copy()
        ranked["_eligible"] = False
        ranked = ranked.sort_values(
            ["quiet_tracks_per_month", "storm_recall", "coverage_000_024", "threshold_score"],
            ascending=[True, False, False, True],
        )
        return ranked.iloc[0], merged, "NO_FULLY_ELIGIBLE_POLICY"
    eligible["_eligible"] = True
    ranked = eligible.sort_values(["storm_recall", "coverage_000_024", "threshold_score"], ascending=[False, False, True])
    return ranked.iloc[0], merged, "OK"


def _baseline_row(sweep_2025: pd.DataFrame, quiet: pd.DataFrame) -> pd.Series:
    """Return the explicit Ops-1 baseline row joined with quiet FAR."""

    base = sweep_2025.loc[sweep_2025["policy_id"] == "ops1_baseline"].merge(
        quiet[["policy_id", "quiet_tracks_per_month"]],
        on="policy_id",
        how="left",
    )
    if base.empty:
        raise RuntimeError("ops1_baseline row missing")
    return base.iloc[0]


def _sensitivity_rows(winner: pd.Series, sweep_2025: pd.DataFrame, quiet: pd.DataFrame) -> pd.DataFrame:
    """Return sensitivity rows toggling t_96_120 between 0.8 and EXCLUDE."""

    rows: list[pd.Series] = []
    for t96 in ("0.8", "EXCLUDE"):
        part = sweep_2025.loc[
            (sweep_2025["t_0_24"] == winner["t_0_24"])
            & (sweep_2025["t_24_48"] == winner["t_24_48"])
            & (sweep_2025["t_48_96"] == winner["t_48_96"])
            & (sweep_2025["t_96_120"].astype(str) == t96)
        ]
        if not part.empty:
            rows.append(part.iloc[0])
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.merge(quiet[["policy_id", "quiet_tracks_per_month"]], on="policy_id", how="left")


def _deployment_config(winner: pd.Series, selection_status: str) -> pd.DataFrame:
    """Build the frozen deployment configuration table."""

    t96 = winner["t_96_120"]
    review_mode = "MANUAL_REVIEW" if selection_status == "OK" else "REVIEW_REQUIRED"
    return pd.DataFrame(
        [
            {"lead_bin": "000-024", "conf_thresh": winner["t_0_24"], "mode": "AUTO" if selection_status == "OK" else "REVIEW_REQUIRED"},
            {"lead_bin": "024-048", "conf_thresh": winner["t_24_48"], "mode": "AUTO" if selection_status == "OK" else "REVIEW_REQUIRED"},
            {"lead_bin": "048-096", "conf_thresh": winner["t_48_96"], "mode": "AUTO" if selection_status == "OK" else "REVIEW_REQUIRED"},
            {
                "lead_bin": "096-120",
                "conf_thresh": t96,
                "mode": review_mode if str(t96).upper() != "EXCLUDE" else "EXCLUDE",
            },
        ]
    )


def _write(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame and print its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-2025", default=str(ROOT / "outputs" / "predictions_2025_sweep.csv"))
    parser.add_argument("--config-2025", default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--split-2025", default="all", choices=["all", "train", "val"])
    parser.add_argument("--predictions-2024val", default=str(ROOT / "outputs" / "predictions_2024val_sweep.csv"))
    parser.add_argument("--config-2024val", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--split-2024val", default="val", choices=["all", "train", "val"])
    parser.add_argument("--predictions-quiet", default=str(ROOT / "outputs" / "predictions_quiet_sweep.csv"))
    parser.add_argument("--quiet-config", default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--hit-threshold-km", type=float, default=60.0)
    parser.add_argument("--track-match-ratio", type=float, default=0.5)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    policies = _policies()
    pred_2025, ref_2025, cfg_2025 = _load_truth_dataset(Path(args.config_2025), Path(args.predictions_2025), args.split_2025, "2025")
    pred_2024, ref_2024, cfg_2024 = _load_truth_dataset(Path(args.config_2024val), Path(args.predictions_2024val), args.split_2024val, "2024val")
    pred_quiet = pd.read_csv(args.predictions_quiet)
    cfg_quiet = load_config(args.quiet_config)

    sweep_2025 = evaluate_policy_dataset(
        dataset="2025",
        predictions=pred_2025,
        references=ref_2025,
        config=cfg_2025,
        policies=policies,
        hit_threshold_km=float(args.hit_threshold_km),
        track_match_ratio=float(args.track_match_ratio),
    )
    sweep_2024 = evaluate_policy_dataset(
        dataset="2024val",
        predictions=pred_2024,
        references=ref_2024,
        config=cfg_2024,
        policies=policies,
        hit_threshold_km=float(args.hit_threshold_km),
        track_match_ratio=float(args.track_match_ratio),
    )
    quiet = evaluate_quiet_policies(predictions=pred_quiet, config=cfg_quiet, policies=policies)

    winner, joined, selection_status = _select_winner(sweep_2025, quiet)
    baseline = _baseline_row(sweep_2025, quiet)
    sensitivity = _sensitivity_rows(winner, sweep_2025, quiet)
    deployment = _deployment_config(winner, selection_status)
    selection = pd.DataFrame(
        [
            {"row": "selected_policy", "selection_status": selection_status, **winner.to_dict()},
            {"row": "ops1_baseline", **baseline.to_dict()},
        ]
    )

    out_dir = Path(args.out_dir)
    _write(out_dir / "ops_policy_sweep_2025.csv", joined)
    _write(out_dir / "ops_policy_sweep_2024val.csv", sweep_2024)
    _write(out_dir / "ops_policy_sweep_quiet.csv", quiet)
    _write(out_dir / "ops_policy_selection.csv", selection)
    _write(out_dir / "ops_policy_96_120_sensitivity.csv", sensitivity)
    _write(out_dir / "ops_deployment_config_frozen.csv", deployment)

    print("Winning policy:")
    print(pd.DataFrame([winner]).to_string(index=False))
    print("Ops-1 baseline:")
    print(pd.DataFrame([baseline]).to_string(index=False))
    print("96-120 sensitivity:")
    print(sensitivity.to_string(index=False))
    print("Frozen deployment config:")
    print(deployment.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
