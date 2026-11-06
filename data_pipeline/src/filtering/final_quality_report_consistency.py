"""Consistency gate for the final eligibility-corrected report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import PROJECT_ROOT, project_path, load_config


def _check(checks: list[dict[str, Any]], name: str, passed: bool, details: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "details": details})


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(project_path(config_path))
    root = project_path(config["outputs"]["root"])
    report_root = project_path(config["outputs"]["report_source"])
    generated = report_root / "generated"
    daily = pd.read_parquet(root / "selected_equity_daily.parquet")
    master = pd.read_parquet(root / "selected_security_master.parquet")
    exclusions = pd.read_csv(root / "historical_exclusions.csv", low_memory=False)
    summary = pd.read_csv(root / "final_data_quality_summary.csv")
    zero = pd.read_csv(root / "remaining_zero_volume_cases.csv", low_memory=False)
    extremes = pd.read_csv(
        root / "remaining_extreme_value_cases.csv", low_memory=False
    )
    cleaning = json.loads(
        (root / "cleaning_policy.json").read_text(encoding="utf-8")
    )
    statistics = json.loads(
        (generated / "report_statistics.json").read_text(encoding="utf-8")
    )
    macros = (generated / "macros.tex").read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []
    snapshot = cleaning["snapshot"]

    _check(checks, "membership_unchanged", len(master) == 488, "488 securities")
    _check(checks, "issuer_count", master["cik"].nunique() == 486, "486 issuers")
    _check(
        checks,
        "final_observation_count",
        len(daily) == snapshot["observations"] == 1186196,
        f"{len(daily)} rows",
    )
    _check(
        checks,
        "security_date_unique",
        not daily.duplicated(["security_id", "date"]).any(),
        "No duplicate security-date keys",
    )
    _check(
        checks,
        "final_return_null",
        daily["pct_return"].isna().all(),
        "pct_return remains null",
    )
    ferg = exclusions[exclusions["ticker"].eq("FERG")]
    dkng = exclusions[exclusions["ticker"].eq("DKNG")]
    _check(checks, "ferg_exclusion_count", len(ferg) == 1601, str(len(ferg)))
    _check(checks, "dkng_exclusion_count", len(dkng) == 189, str(len(dkng)))
    _check(
        checks,
        "only_two_securities_excluded",
        set(exclusions["security_id"]) == set(cleaning["policy"][i]["security_id"] for i in range(2)),
        ",".join(sorted(set(exclusions["ticker"]))),
    )
    _check(
        checks,
        "ferg_boundary",
        ferg["date"].max() == "2022-05-11"
        and daily.loc[
            daily["security_id"].eq(ferg["security_id"].iloc[0]), "date"
        ].min()
        == pd.Timestamp("2022-05-12"),
        "2022-05-11 excluded / 2022-05-12 retained",
    )
    _check(
        checks,
        "dkng_boundary",
        dkng["date"].max() == "2020-04-23"
        and daily.loc[
            daily["security_id"].eq(dkng["security_id"].iloc[0]), "date"
        ].min()
        == pd.Timestamp("2020-04-24"),
        "2020-04-23 excluded / 2020-04-24 retained",
    )
    _check(
        checks,
        "zero_volume_reconciles",
        len(zero) == int(daily["volume"].eq(0).sum()) == snapshot["zero_volume_rows"] == 16,
        f"{len(zero)} rows",
    )
    _check(
        checks,
        "extreme_volume_reconciles",
        int(extremes["extreme_volume_spike_flag"].sum())
        == snapshot["extreme_volume_rows"]
        == 235,
        f"{int(extremes['extreme_volume_spike_flag'].sum())} rows",
    )
    _check(
        checks,
        "extreme_return_reconciles",
        int(extremes["extreme_return_flag"].sum())
        == snapshot["extreme_return_rows"]
        == 45,
        f"{int(extremes['extreme_return_flag'].sum())} rows",
    )
    _check(
        checks,
        "extreme_price_reconciles",
        int(extremes["extreme_price_flag"].sum()) == 0,
        "0 rows",
    )
    _check(
        checks,
        "summary_final_rows",
        int(
            summary.loc[
                summary["metric"].eq("Validated stock-day observations"),
                "final_research_dataset",
            ].iloc[0]
        )
        == len(daily),
        "summary and panel agree",
    )
    _check(
        checks,
        "summary_ferg",
        int(
            summary.loc[
                summary["metric"].eq("FERG observations excluded"),
                "final_research_dataset",
            ].iloc[0]
        )
        == len(ferg),
        "summary and exclusion audit agree",
    )
    _check(
        checks,
        "summary_dkng",
        int(
            summary.loc[
                summary["metric"].eq("DKNG observations excluded"),
                "final_research_dataset",
            ].iloc[0]
        )
        == len(dkng),
        "summary and exclusion audit agree",
    )
    _check(
        checks,
        "statistics_snapshot",
        statistics["snapshot"] == snapshot,
        "Report statistics use cleaning-policy snapshot",
    )
    for macro, value in [
        ("FinalSecurities", "488"),
        ("FinalObservations", "1,186,196"),
        ("FergExcluded", "1,601"),
        ("DkngExcluded", "189"),
        ("FinalZeroVolumeRows", "16"),
        ("FinalExtremeVolumeRows", "235"),
        ("FinalExtremeReturnRows", "45"),
    ]:
        _check(
            checks,
            f"macro_{macro}",
            rf"\newcommand{{\{macro}}}{{{value}}}" in macros,
            value,
        )
    for table in [
        "table_eligibility_policy.tex",
        "table_historical_exclusions.tex",
        "table_final_years.tex",
        "table_remaining_zero.tex",
        "table_extreme_summary.tex",
        "table_quality_comparison.tex",
        "table_treatments.tex",
    ]:
        _check(
            checks,
            f"asset_{table}",
            (generated / table).exists(),
            str(generated / table),
        )
    _check(
        checks,
        "protected_sources",
        cleaning["research_constraints"]["modify_source_data"] is False,
        "Protected archives are read-only by policy",
    )
    result = {
        "passed": all(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_failed": sum(not check["passed"] for check in checks),
        "snapshot": snapshot,
    }
    (report_root / "generated").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(
        generated / "report_consistency_checks.csv", index=False
    )
    (generated / "report_consistency_summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "historical_eligibility.yaml"),
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
