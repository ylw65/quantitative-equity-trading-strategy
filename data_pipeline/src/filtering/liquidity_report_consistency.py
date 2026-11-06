"""Fail when the liquidity report disagrees with its data snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common import PROJECT_ROOT, atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _macro_values(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}", text
        )
    }


def _as_int(value: str) -> int:
    return int(value.replace(",", ""))


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def check_consistency(
    root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    filtered = root / "data/filtered/adtv100m_2023"
    quality = filtered / "quality"
    report = root / "reports/liquidity_report"

    config = yaml.safe_load(
        (root / "configs/liquidity_filter.yaml").read_text(encoding="utf-8")
    )
    metadata = json.loads((filtered / "filter_metadata.json").read_text())
    integrity = json.loads(
        (quality / "post_filter_integrity_summary.json").read_text()
    )
    anomalies = json.loads(
        (quality / "post_filter_anomaly_summary.json").read_text()
    )
    stats = json.loads(
        (report / "generated/report_statistics.json").read_text()
    )
    macros = _macro_values(report / "generated/macros.tex")

    master = pd.read_parquet(
        filtered / "selected_security_master.parquet",
        columns=["security_id", "cik"],
    )
    daily = pd.read_parquet(
        filtered / "selected_equity_daily.parquet",
        columns=["security_id", "date", "pct_return"],
    )
    screen = pd.read_csv(
        filtered / "liquidity_screen_all.csv", low_memory=False
    )
    digest = screen.loc[screen["selected"].fillna(False)].copy()
    action_classes = pd.read_csv(
        root / "reports/tables/liquidity/postfilter_corporate_action_classes.csv"
    )

    selected_ids = set(master["security_id"].astype(str))
    daily_ids = set(daily["security_id"].astype(str))
    expected_hashes = metadata["source_hashes_sha256"]
    current_hashes = {
        name: _sha256(root / path)
        for name, path in config["inputs"].items()
    }
    title = "Liquidity-Based Universe Selection and Post-Filter Data Quality Assessment"
    main_tex = (report / "main.tex").read_text(encoding="utf-8")
    combined_report_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [report / "main.tex", *sorted((report / "sections").glob("*.tex"))]
    )
    normalized_report_text = re.sub(
        r"\s+", " ", combined_report_text.replace("\\\\", " ")
    )

    checks: list[tuple[str, Any, Any]] = [
        ("universe_count", len(screen), metadata["universe_securities"]),
        (
            "acquired_count",
            int(screen["acquisition_status"].eq("acquired").sum()),
            metadata["acquired_securities"],
        ),
        ("screen_rows_unique", bool(screen["security_id"].is_unique), True),
        ("selected_screen_count", len(digest), metadata["selected_securities"]),
        ("selected_master_count", len(master), metadata["selected_securities"]),
        ("selected_daily_rows", len(daily), metadata["selected_observations"]),
        ("selected_ids_reconcile", selected_ids, daily_ids),
        ("selected_screen_ids_reconcile", set(digest["security_id"].astype(str)), selected_ids),
        (
            "selected_threshold_inclusive",
            bool(digest["adtv20_usd"].ge(config["project"]["threshold_usd"]).all()),
            True,
        ),
        ("selected_issuers", int(master["cik"].nunique()), metadata["selected_distinct_issuers"]),
        (
            "selected_security_date_unique",
            bool(not daily.duplicated(["security_id", "date"]).any()),
            True,
        ),
        ("pct_return_null", bool(daily["pct_return"].isna().all()), True),
        (
            "daily_date_range",
            [
                pd.Timestamp(daily["date"].min()).date().isoformat(),
                pd.Timestamp(daily["date"].max()).date().isoformat(),
            ],
            ["2016-01-04", "2025-12-31"],
        ),
        ("window_session_count", len(metadata["window_sessions"]), 20),
        ("window_first_session", metadata["window_sessions"][0], "2022-12-02"),
        ("window_last_session", metadata["window_sessions"][-1], "2022-12-30"),
        (
            "integrity_selected_count",
            integrity["selected_securities"],
            len(master),
        ),
        (
            "integrity_selected_rows",
            integrity["validated_rows"],
            len(daily),
        ),
        (
            "identity_metric_uses_review_language",
            "identity_review_candidates" in integrity
            and "known_contamination_securities" not in integrity,
            True,
        ),
        (
            "identity_review_tickers",
            integrity["identity_review_candidate_tickers"],
            ["ABNB", "AFRM"],
        ),
        (
            "anomaly_selected_count",
            anomalies["selected_securities"],
            len(master),
        ),
        (
            "anomaly_rows_reconcile",
            anomalies["selected_validated_rows"],
            len(daily),
        ),
        (
            "source_hashes_unchanged",
            current_hashes,
            expected_hashes,
        ),
        (
            "protected_inputs_unchanged",
            integrity["protected_inputs_unchanged"],
            True,
        ),
        ("anomaly_source_unmodified", anomalies["source_data_modified"], False),
        (
            "report_statistics_selected",
            stats["snapshot"]["selected_securities"],
            len(master),
        ),
        (
            "report_statistics_rows",
            stats["snapshot"]["validated_rows"],
            len(daily),
        ),
        (
            "macro_selected",
            _as_int(macros["SelectedSecurities"]),
            len(master),
        ),
        (
            "macro_observations",
            _as_int(macros["SelectedObservations"]),
            len(daily),
        ),
        (
            "macro_raw_rows",
            _as_int(macros["RawSelectedRows"]),
            integrity["raw_rows"],
        ),
        (
            "macro_zero_volume",
            _as_int(macros["ZeroVolumeRows"]),
            integrity["zero_volume_rows"],
        ),
        (
            "macro_extreme_returns",
            _as_int(macros["ExtremeReturns"]),
            integrity["extreme_return_rows"],
        ),
        (
            "macro_internal_gaps",
            _as_int(macros["InternalCalendarGaps"]),
            integrity["internal_calendar_gaps"],
        ),
        (
            "macro_identity_candidates",
            _as_int(macros["IdentityReviewCandidates"]),
            integrity["identity_review_candidates"],
        ),
        (
            "macro_corporate_action_cases",
            _as_int(macros["CorporateActionCaseRecords"]),
            int(action_classes["events"].sum()),
        ),
        ("report_title_present", title in normalized_report_text, True),
        (
            "main_uses_generated_macros",
            "\\GeneratedInput{macros}" in main_tex,
            True,
        ),
        (
            "generated_assets_present",
            {
                path.name
                for path in [
                    *(report / "generated").glob("table_*.tex"),
                    *(report / "generated").glob("*.json"),
                    *(root / "reports/figures/liquidity").glob("liquidity_*.pdf"),
                ]
            },
            {
                "table_before_after.tex",
                "table_selection_flow.tex",
                "table_composition.tex",
                "report_statistics.json",
                "liquidity_adtv20_ecdf.pdf",
                "liquidity_quality_by_year.pdf",
            },
        ),
        (
            "no_inexact_contamination_language",
            "known contamination" not in combined_report_text.lower(),
            True,
        ),
    ]

    if "generated_assets_present" in [row[0] for row in checks]:
        index = [row[0] for row in checks].index("generated_assets_present")
        name, actual, required = checks[index]
        checks[index] = (name, required.issubset(actual), True)

    rows = [
        {
            "check": name,
            "actual": json.dumps(_json_safe(actual), sort_keys=True)
            if isinstance(actual, (dict, list, set))
            else actual,
            "expected": json.dumps(_json_safe(expected), sort_keys=True)
            if isinstance(expected, (dict, list, set))
            else expected,
            "passed": actual == expected,
        }
        for name, actual, expected in checks
    ]
    frame = pd.DataFrame(rows)
    payload = {
        "passed": bool(frame["passed"].all()),
        "checks_total": len(frame),
        "checks_passed": int(frame["passed"].sum()),
        "checks_failed": int((~frame["passed"]).sum()),
        "snapshot": {
            "selected_securities": len(master),
            "selected_validated_rows": len(daily),
            "raw_selected_rows": integrity["raw_rows"],
        },
    }
    return payload, frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    payload, checks = check_consistency(args.root)
    output = args.root / "data/filtered/adtv100m_2023/quality"
    output.mkdir(parents=True, exist_ok=True)
    checks.to_csv(output / "report_consistency_checks.csv", index=False)
    atomic_write_json(payload, output / "report_consistency_summary.json")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        print(checks.loc[~checks["passed"]].to_string(index=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
