"""Build project-format IBTrACS truth with overlap consistency gates."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401

import numpy as np
import pandas as pd

from tclocator.common import DomainConfig, haversine_km, load_config


LAST3_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.last3years.list.v04r01.csv"
)
ALL_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
REQUIRED_COLUMNS = ["SID", "ISO_TIME", "LAT", "LON"]
OPTIONAL_FILTER_COLUMNS = ["TRACK_TYPE", "NATURE"]
OVERLAP_MONTHS = [f"2024-{month:02d}" for month in range(5, 12)]


def _audit_dir(config: dict[str, Any]) -> Path:
    """Return the audit output directory."""

    out = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs")) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _download(url: str, target: Path) -> bool:
    """Download one URL to ``target`` and return whether it succeeded."""

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Download failed: {url} ({type(exc).__name__}: {exc})")
        return False


def _resolve_raw_source(config: dict[str, Any], input_path: str | None) -> Path:
    """Resolve or download the raw IBTrACS CSV."""

    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    raw_dir = Path(config.get("paths", {}).get("ibtracs_raw_dir", ROOT / "data" / "ibtracs" / "raw"))
    last3 = raw_dir / Path(LAST3_URL).name
    if last3.exists():
        return last3
    if _download(LAST3_URL, last3):
        return last3

    all_path = raw_dir / Path(ALL_URL).name
    if all_path.exists():
        return all_path
    if _download(ALL_URL, all_path):
        return all_path

    manual = (
        "IBTrACS download failed. Manually download one of these files and rerun with --input:\n"
        f"  {LAST3_URL}\n"
        f"  {ALL_URL}\n"
        f"Place it under: {raw_dir}"
    )
    raise RuntimeError(manual)


def _read_raw_ibtracs(path: Path) -> pd.DataFrame:
    """Read the needed columns from an official IBTrACS CSV."""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = [col for col in REQUIRED_COLUMNS + OPTIONAL_FILTER_COLUMNS if col in header]
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"IBTrACS raw CSV is missing required columns {missing}: {path}")
    return pd.read_csv(path, skiprows=[1], usecols=columns, low_memory=False)


def _time_format_from_old(old_truth: pd.DataFrame) -> str:
    """Infer the project ISO_TIME string format from the existing truth file."""

    sample = str(old_truth["ISO_TIME"].dropna().astype(str).iloc[0])
    if "T" in sample:
        return "%Y-%m-%dT%H:%M:%S"
    return "%Y-%m-%d %H:%M:%S"


def _convert_raw(raw: pd.DataFrame, domain: DomainConfig, time_format: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert raw IBTrACS rows to the four-column project georef contract."""

    stats: dict[str, int] = {"raw_rows": int(len(raw))}
    df = raw.copy()
    stats["after_track_type_filter"] = int(len(df))

    time = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    lat = pd.to_numeric(df["LAT"], errors="coerce")
    lon = pd.to_numeric(df["LON"], errors="coerce")
    valid_numeric = time.notna() & lat.notna() & lon.notna()
    stats["dropped_bad_numeric_or_time"] = int((~valid_numeric).sum())
    df = df.loc[valid_numeric].copy()
    time = time.loc[valid_numeric]
    lat = lat.loc[valid_numeric].astype(float)
    lon = np.mod(lon.loc[valid_numeric].astype(float), 360.0)

    six_hour = time.dt.hour.isin([0, 6, 12, 18]) & time.dt.minute.eq(0)
    stats["dropped_non_6h"] = int((~six_hour).sum())
    df = df.loc[six_hour].copy()
    time = time.loc[six_hour]
    lat = lat.loc[six_hour]
    lon = lon.loc[six_hour]

    in_domain = lat.between(domain.lat_min, domain.lat_max) & lon.between(domain.lon_min, domain.lon_max)
    stats["dropped_out_of_domain"] = int((~in_domain).sum())
    df = df.loc[in_domain].copy()
    time = time.loc[in_domain]
    lat = lat.loc[in_domain]
    lon = lon.loc[in_domain]

    out = pd.DataFrame(
        {
            "ISO_TIME": time.dt.strftime(time_format),
            "SID": df["SID"].astype(str).str.strip(),
            "LAT": lat.astype(float),
            "LON": lon.astype(float),
        }
    )
    before = len(out)
    out = out.drop_duplicates(subset=["ISO_TIME", "SID"], keep="first")
    stats["dropped_duplicate_time_sid"] = int(before - len(out))
    if (out["LON"] < 0).any():
        raise ValueError("Converted truth contains negative longitude after 0-360 conversion")
    return out.sort_values(["ISO_TIME", "SID"]).reset_index(drop=True), stats


def _project_old_truth(path: Path) -> pd.DataFrame:
    """Read only the four columns consumed by the project pipeline."""

    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    df["ISO_TIME"] = df["ISO_TIME"].astype(str)
    df["SID"] = df["SID"].astype(str)
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = np.mod(pd.to_numeric(df["LON"], errors="coerce"), 360.0)
    return df.dropna(subset=["ISO_TIME", "SID", "LAT", "LON"]).reset_index(drop=True)


def _month_counts(df: pd.DataFrame) -> pd.Series:
    """Return monthly record counts keyed by ``YYYY-MM``."""

    time = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    return time.dt.strftime("%Y-%m").value_counts().sort_index()


def _nearest_same_time_distances(new_rows: pd.DataFrame, old_rows: pd.DataFrame) -> pd.DataFrame:
    """Fallback consistency rows when SID systems do not overlap."""

    rows: list[dict[str, Any]] = []
    for _, old in old_rows.iterrows():
        same_time = new_rows.loc[new_rows["ISO_TIME"] == old["ISO_TIME"]]
        if same_time.empty:
            rows.append({**old.to_dict(), "matched_sid": "", "dist_km": np.nan, "match_mode": "missing_time"})
            continue
        dist = haversine_km(old["LAT"], old["LON"], same_time["LAT"].to_numpy(), same_time["LON"].to_numpy())
        idx = int(np.nanargmin(dist))
        nearest = same_time.iloc[idx]
        rows.append(
            {
                "ISO_TIME": old["ISO_TIME"],
                "SID": old["SID"],
                "LAT": old["LAT"],
                "LON": old["LON"],
                "matched_sid": nearest["SID"],
                "matched_lat": nearest["LAT"],
                "matched_lon": nearest["LON"],
                "dist_km": float(np.asarray(dist)[idx]),
                "match_mode": "nearest_same_time",
            }
        )
    return pd.DataFrame(rows)


def _overlap_gate(old_truth: pd.DataFrame, new_truth: pd.DataFrame, out_dir: Path) -> bool:
    """Run the 2024 overlap consistency gate."""

    old = old_truth.copy()
    new = new_truth.copy()
    old["year_month"] = pd.to_datetime(old["ISO_TIME"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    new["year_month"] = pd.to_datetime(new["ISO_TIME"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    mismatch_rows: list[pd.DataFrame] = []

    for month in OVERLAP_MONTHS:
        old_month = old.loc[old["year_month"] == month, REQUIRED_COLUMNS].copy()
        new_month = new.loc[new["year_month"] == month, REQUIRED_COLUMNS].copy()
        ratio = float(len(new_month) / len(old_month)) if len(old_month) else np.nan
        merged = old_month.merge(new_month, on=["ISO_TIME", "SID"], how="left", suffixes=("_old", "_new"))
        matched = merged.dropna(subset=["LAT_new", "LON_new"])
        if len(matched) > 0:
            distances = haversine_km(
                matched["LAT_old"].to_numpy(),
                matched["LON_old"].to_numpy(),
                matched["LAT_new"].to_numpy(),
                matched["LON_new"].to_numpy(),
            )
            median_dist = float(np.nanmedian(distances))
            max_dist = float(np.nanmax(distances))
            match_mode = "time_sid"
        else:
            nearest = _nearest_same_time_distances(new_month, old_month)
            distances = nearest["dist_km"].to_numpy(dtype=float) if not nearest.empty else np.array([np.nan])
            median_dist = float(np.nanmedian(distances))
            max_dist = float(np.nanmax(distances))
            match_mode = "nearest_same_time"
            mismatch_rows.append(nearest.head(20))
        rows.append(
            {
                "year_month": month,
                "n_records_old": int(len(old_month)),
                "n_records_new": int(len(new_month)),
                "new_old_ratio": ratio,
                "matched_records": int(len(matched)),
                "position_median_km": median_dist,
                "position_max_km": max_dist,
                "match_mode": match_mode,
                "pass": bool(0.9 <= ratio <= 1.1 and median_dist < 30.0),
            }
        )
        bad = merged.loc[merged["LAT_new"].isna()].head(20)
        if not bad.empty:
            mismatch_rows.append(bad)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "truth_overlap_consistency.csv", index=False)
    if mismatch_rows:
        pd.concat(mismatch_rows, ignore_index=True).to_csv(out_dir / "truth_mismatch.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "truth_mismatch.csv", index=False)
    return bool(summary["pass"].all())


def _merge_truth(old_truth: pd.DataFrame, new_truth: pd.DataFrame) -> pd.DataFrame:
    """Merge old truth with new source rows, keeping old rows authoritative."""

    old = old_truth[REQUIRED_COLUMNS].copy()
    new = new_truth[REQUIRED_COLUMNS].copy()
    old_keys = set(zip(old["ISO_TIME"].astype(str), old["SID"].astype(str), strict=True))
    new["year"] = pd.to_datetime(new["ISO_TIME"], utc=True, errors="coerce").dt.year
    additions = new.loc[
        new["year"].between(2020, 2025)
        & ~new.apply(lambda row: (str(row["ISO_TIME"]), str(row["SID"])) in old_keys, axis=1)
    ]
    merged = pd.concat([old, additions[REQUIRED_COLUMNS]], ignore_index=True)
    return merged.drop_duplicates(subset=["ISO_TIME", "SID"], keep="first").sort_values(["ISO_TIME", "SID"]).reset_index(drop=True)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--input", default=None, help="Optional existing official IBTrACS raw CSV")
    parser.add_argument("--old-georef", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    domain = DomainConfig.from_mapping(config.get("domain"))
    out_dir = _audit_dir(config)
    old_path = Path(args.old_georef or ROOT / "data" / "ibtracs" / "georef.csv")
    output_path = Path(args.output or ROOT / "data" / "ibtracs" / "georef_2020_2025.csv")
    if not old_path.exists():
        print(f"Existing georef not found: {old_path}")
        return 1

    try:
        raw_path = _resolve_raw_source(config, args.input)
    except Exception as exc:
        print(str(exc))
        return 1

    old_truth = _project_old_truth(old_path)
    time_format = _time_format_from_old(old_truth)
    raw = _read_raw_ibtracs(raw_path)
    converted, conversion_stats = _convert_raw(raw, domain, time_format)
    converted.to_csv(out_dir / "ibtracs_converted_project_format.csv", index=False)
    (out_dir / "ibtracs_conversion_stats.json").write_text(
        pd.Series(conversion_stats).to_json(indent=2),
        encoding="utf-8",
    )

    pass_gate = _overlap_gate(old_truth, converted, out_dir)
    if not pass_gate:
        print(f"FAIL overlap consistency. See {out_dir / 'truth_overlap_consistency.csv'} and truth_mismatch.csv")
        return 2

    merged = _merge_truth(old_truth, converted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    counts = _month_counts(merged)
    counts.to_csv(out_dir / "georef_2020_2025_monthly_counts.csv", header=["n_records"])
    print(f"Raw source: {raw_path}")
    print(f"Wrote {out_dir / 'ibtracs_converted_project_format.csv'}")
    print(f"Wrote {out_dir / 'truth_overlap_consistency.csv'}")
    print(f"Wrote {output_path}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
