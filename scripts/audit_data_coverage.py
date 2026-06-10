"""Audit AIFS inventory, IBTrACS truth coverage, and monthly alignment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tclocator import _pygrib as _pygrib  # noqa: F401
from tclocator.io_aifs import AIFSFileMeta, parse_aifs_filename, read_aifs_channels
import numpy as np
import pandas as pd

from tclocator.common import DomainConfig, build_lat_lon, haversine_km, in_domain, iter_files, latlon_to_grid, load_config
from tclocator.labels import read_ibtracs, records_at_time


@dataclass(frozen=True)
class ParsedAIFS:
    """One parseable AIFS file."""

    path: Path
    meta: AIFSFileMeta

    @property
    def init_month(self) -> str:
        """Return initialization month as ``YYYY-MM``."""

        return self.meta.init_time.strftime("%Y-%m")


def _audit_dir(config: dict[str, Any]) -> Path:
    """Return the audit output directory."""

    out = Path(config.get("paths", {}).get("output_dir", ROOT / "outputs")) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _parse_aifs_files(config: dict[str, Any]) -> tuple[list[ParsedAIFS], pd.DataFrame]:
    """Scan and parse all AIFS files under the configured root."""

    files = iter_files(config.get("paths", {}).get("aifs_dir", ""), [".pt", ".grib2", ".grb2", ".grib"])
    parsed: list[ParsedAIFS] = []
    bad_rows: list[dict[str, str]] = []
    for path in files:
        try:
            parsed.append(ParsedAIFS(path=path, meta=parse_aifs_filename(path)))
        except Exception as exc:
            bad_rows.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return parsed, pd.DataFrame(bad_rows)


def _inventory(parsed: list[ParsedAIFS]) -> pd.DataFrame:
    """Summarize AIFS inventory by initialization month."""

    rows: list[dict[str, Any]] = []
    for month in sorted({item.init_month for item in parsed}):
        items = [item for item in parsed if item.init_month == month]
        leads_by_init: dict[str, list[int]] = defaultdict(list)
        for item in items:
            leads_by_init[item.meta.init_time.isoformat()].append(int(item.meta.forecast_hour))
        lead_counts = [len(set(leads)) for leads in leads_by_init.values()]
        missing_examples: list[str] = []
        for init_time, leads in sorted(leads_by_init.items())[:20]:
            unique = sorted(set(leads))
            if not unique:
                continue
            expected = set(range(min(unique), max(unique) + 1, 6))
            missing = sorted(expected.difference(unique))
            if missing:
                missing_examples.append(f"{init_time}:missing={missing[:8]}")
            if len(missing_examples) >= 5:
                break
        all_leads = [int(item.meta.forecast_hour) for item in items]
        rows.append(
            {
                "year_month": month,
                "n_files": len(items),
                "n_inits": len(leads_by_init),
                "lead_min": min(all_leads) if all_leads else np.nan,
                "lead_max": max(all_leads) if all_leads else np.nan,
                "n_leads_per_init_median": float(np.median(lead_counts)) if lead_counts else np.nan,
                "missing_leads_examples": "; ".join(missing_examples),
            }
        )
    return pd.DataFrame(rows)


def _ibtracs_coverage(config: dict[str, Any]) -> pd.DataFrame:
    """Summarize IBTrACS truth records by month."""

    path = Path(config.get("paths", {}).get("ibtracs_csv", ""))
    if not path.exists():
        return pd.DataFrame(columns=["year_month", "n_records", "n_sids"])
    records = read_ibtracs(path, config.get("ibtracs", {}).get("col_map", {}))
    if records.empty:
        return pd.DataFrame(columns=["year_month", "n_records", "n_sids"])
    records = records.copy()
    records["year_month"] = records["ISO_TIME"].dt.strftime("%Y-%m")
    return (
        records.groupby("year_month")
        .agg(n_records=("ISO_TIME", "size"), n_sids=("SID", "nunique"))
        .reset_index()
        .sort_values("year_month")
    )


def _truth_table(inventory: pd.DataFrame, ib_coverage: pd.DataFrame) -> pd.DataFrame:
    """Join AIFS inventory with truth coverage."""

    if inventory.empty:
        return pd.DataFrame(columns=["year_month", "n_files", "n_inits", "truth"])
    truth_months = set(ib_coverage["year_month"].astype(str)) if not ib_coverage.empty else set()
    out = inventory.copy()
    out["truth"] = out["year_month"].map(lambda month: "OK" if str(month) in truth_months else "MISSING")
    out = out.merge(ib_coverage, on="year_month", how="left")
    out["n_records"] = out["n_records"].fillna(0).astype(int)
    out["n_sids"] = out["n_sids"].fillna(0).astype(int)
    return out


def _nearest_truth_value(msl: np.ndarray, lat: float, lon: float, domain: DomainConfig) -> float:
    """Return MSL at the nearest truth grid point."""

    y_f, x_f = latlon_to_grid(lat, lon, domain, clip=True)
    y = int(np.clip(round(float(y_f)), 0, domain.height - 1))
    x = int(np.clip(round(float(x_f)), 0, domain.width - 1))
    return float(msl[y, x])


def _local_min_100km(msl: np.ndarray, lat: float, lon: float, domain: DomainConfig) -> tuple[float, float]:
    """Return local minimum MSL and distance to truth inside 100 km."""

    lat1d, lon1d = build_lat_lon(domain)
    dist = haversine_km(lat1d[:, None], lon1d[None, :], lat, lon)
    mask = dist <= 100.0
    if not np.any(mask):
        return float("nan"), float("nan")
    masked = np.where(mask, msl, np.inf)
    y, x = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return float(masked[y, x]), float(dist[y, x])


def _candidate_cases(parsed: list[ParsedAIFS], records: pd.DataFrame, months: list[str], max_cases_per_month: int) -> dict[str, list[tuple[ParsedAIFS, pd.Series]]]:
    """Pick short-lead monthly alignment candidates with matching truth."""

    by_month: dict[str, list[tuple[ParsedAIFS, pd.Series]]] = {}
    for month in months:
        month_items = sorted(
            [item for item in parsed if item.init_month == month and item.meta.forecast_hour <= 24],
            key=lambda item: (item.meta.forecast_hour, item.meta.valid_time, str(item.path)),
        )
        cases: list[tuple[ParsedAIFS, pd.Series]] = []
        for item in month_items:
            at_time = records_at_time(records, pd.Timestamp(item.meta.valid_time))
            if at_time.empty:
                continue
            for _, record in at_time.iterrows():
                cases.append((item, record))
                if len(cases) >= max_cases_per_month:
                    break
            if len(cases) >= max_cases_per_month:
                break
        by_month[month] = cases
    return by_month


def _alignment_audit(
    config: dict[str, Any],
    parsed: list[ParsedAIFS],
    usable_months: list[str],
    *,
    max_cases_per_month: int,
) -> pd.DataFrame:
    """Run monthly production-reader alignment checks."""

    ib_path = Path(config.get("paths", {}).get("ibtracs_csv", ""))
    if not ib_path.exists() or not usable_months:
        return pd.DataFrame()
    records = read_ibtracs(ib_path, config.get("ibtracs", {}).get("col_map", {}))
    domain = DomainConfig.from_mapping(config.get("domain"))
    rows: list[dict[str, Any]] = []
    candidates = _candidate_cases(parsed, records, usable_months, max_cases_per_month)
    for month in usable_months:
        cases = candidates.get(month, [])
        if not cases:
            rows.append({"year_month": month, "status": "NO_MATCHING_SHORT_LEAD_TRUTH"})
            continue
        for item, record in cases:
            field, _ = read_aifs_channels(item.path, channels=["msl"], domain=domain, aifs_config=config.get("aifs", {}))
            lat = float(record["LAT"])
            lon = float(record["LON"])
            if not in_domain(lat, lon, domain):
                rows.append(
                    {
                        "year_month": month,
                        "path": str(item.path),
                        "sid": str(record["SID"]),
                        "valid_time": item.meta.valid_time.isoformat(),
                        "lead_hour": int(item.meta.forecast_hour),
                        "truth_lat": lat,
                        "truth_lon": lon,
                        "status": "TRUTH_OUT_OF_DOMAIN",
                        "strict_pass": False,
                        "relaxed_pass": False,
                    }
                )
                continue
            truth_pa = _nearest_truth_value(field[0], lat, lon, domain)
            min_pa, min_dist_km = _local_min_100km(field[0], lat, lon, domain)
            strict_pass = truth_pa < 99_500.0 and min_pa < 99_000.0
            relaxed_pass = truth_pa < 100_000.0 or min_pa < 100_000.0
            rows.append(
                {
                    "year_month": month,
                    "path": str(item.path),
                    "sid": str(record["SID"]),
                    "valid_time": item.meta.valid_time.isoformat(),
                    "lead_hour": int(item.meta.forecast_hour),
                    "truth_lat": lat,
                    "truth_lon": lon,
                    "msl_truth_hPa": truth_pa / 100.0,
                    "min100_hPa": min_pa / 100.0,
                    "min100_dist_km": min_dist_km,
                    "strict_pass": bool(strict_pass),
                    "relaxed_pass": bool(relaxed_pass),
                    "status": "PASS" if strict_pass else ("RELAXED_PASS" if relaxed_pass else "FAIL"),
                }
            )
            if relaxed_pass:
                break
    return pd.DataFrame(rows)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "finetune.yaml"))
    parser.add_argument("--max-cases-per-month", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = _audit_dir(config)
    parsed, unparseable = _parse_aifs_files(config)
    inventory = _inventory(parsed)
    ib_coverage = _ibtracs_coverage(config)
    truth = _truth_table(inventory, ib_coverage)

    inventory.to_csv(out_dir / "aifs_inventory.csv", index=False)
    unparseable.to_csv(out_dir / "unparseable.csv", index=False)
    ib_coverage.to_csv(out_dir / "ibtracs_coverage.csv", index=False)
    truth.to_csv(out_dir / "aifs_truth_join.csv", index=False)

    truth_usable_months = truth.loc[truth["truth"] == "OK", "year_month"].astype(str).tolist() if not truth.empty else []
    truth_blocked_months = truth.loc[truth["truth"] == "MISSING", "year_month"].astype(str).tolist() if not truth.empty else []
    alignment = _alignment_audit(config, parsed, truth_usable_months, max_cases_per_month=max(1, int(args.max_cases_per_month)))
    alignment.to_csv(out_dir / "monthly_alignment_check.csv", index=False)

    alignment_pass_months: list[str] = []
    alignment_blocked_months: list[str] = []
    if truth_usable_months:
        if alignment.empty or "relaxed_pass" not in alignment.columns:
            alignment_blocked_months = list(truth_usable_months)
        else:
            for month in truth_usable_months:
                month_rows = alignment.loc[alignment["year_month"].astype(str) == month]
                passed = bool(month_rows["relaxed_pass"].fillna(False).any())
                if not passed:
                    alignment_blocked_months.append(month)
                else:
                    alignment_pass_months.append(month)
    blocked_months = sorted(set(truth_blocked_months).union(alignment_blocked_months))
    decision = {
        "truth_usable_months": truth_usable_months,
        "truth_blocked_months": truth_blocked_months,
        "alignment_pass_months": alignment_pass_months,
        "alignment_blocked_months": alignment_blocked_months,
        "blocked_months": blocked_months,
        "recommended_data_usable_months": alignment_pass_months,
    }
    decision_path = out_dir / "coverage_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(f"Wrote {out_dir / 'aifs_inventory.csv'}")
    print(f"Wrote {out_dir / 'unparseable.csv'}")
    print(f"Wrote {out_dir / 'ibtracs_coverage.csv'}")
    print(f"Wrote {out_dir / 'aifs_truth_join.csv'}")
    print(f"Wrote {out_dir / 'monthly_alignment_check.csv'}")
    print(f"Wrote {decision_path}")
    if len(unparseable) > 0:
        print(f"WARNING unparseable AIFS files: {len(unparseable)}")
    print("AIFS/truth coverage:")
    print(truth.to_string(index=False) if not truth.empty else "no AIFS inventory")
    print(f"truth_usable_months={truth_usable_months}")
    print(f"alignment_pass_months={alignment_pass_months}")
    print(f"alignment_blocked_months={alignment_blocked_months}")
    print(f"blocked_months={blocked_months}")
    print(f"recommended data.usable_months={alignment_pass_months}")
    if not alignment_pass_months and truth_usable_months:
        print("FAIL")
        return 2
    print("PASS" if not alignment_blocked_months else "PASS_WITH_BLOCKED_MONTHS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
