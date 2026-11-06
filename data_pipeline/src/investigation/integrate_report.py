"""Build compact report artifacts from the P0/P1 investigation evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT


GEN_DIR = PROJECT_ROOT / "reports" / "latex" / "generated"
INVESTIGATION_DIR = PROJECT_ROOT / "data" / "investigation"
ISSUE_DIR = PROJECT_ROOT / "data" / "issues"


def _tex(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "Unknown"
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        return str(integer) if 1900 <= integer <= 2100 else f"{integer:,}"
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) < 10 and float(value) != int(float(value)):
            return f"{float(value):.4f}"
        return f"{float(value):,.2f}"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "_": r"\_\allowbreak{}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _table(
    frame: pd.DataFrame,
    name: str,
    *,
    widths: str | None = None,
    small: bool = False,
    longtable: bool = False,
) -> None:
    columns = list(frame.columns)
    spec = widths or ("l" + "r" * (len(columns) - 1))
    environment = "longtable" if longtable else "tabular"
    lines = [
        rf"\begin{{{environment}}}{{@{{}}{spec}@{{}}}}",
        r"\toprule",
        " & ".join(_tex(column.replace("_", " ")) for column in columns)
        + r" \\",
        r"\midrule",
    ]
    if longtable:
        lines.extend(
            [
                r"\endfirsthead",
                r"\toprule",
                " & ".join(_tex(column.replace("_", " ")) for column in columns)
                + r" \\",
                r"\midrule",
                r"\endhead",
            ]
        )
    for values in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_tex(value) for value in values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            rf"\end{{{environment}}}",
        ]
    )
    content = "\n".join(lines)
    if small:
        content = "{\\small\n" + content + "\n}"
    _write(GEN_DIR / name, content)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _failure_categories() -> pd.DataFrame:
    frame = pd.read_csv(
        INVESTIGATION_DIR / "missing" / "missing_failure_categories.csv"
    )
    labels = {
        "provider_reports_symbol_missing_without_current_match": (
            "No current exact SEC ticker"
        ),
        "ticker_change_alias_unresolved": "SEC-evidenced later symbol",
        "provider_no_history_for_current_listing": (
            "Current SEC ticker, no Yahoo history"
        ),
    }
    frame["Evidence category"] = frame["evidence_category"].map(labels)
    frame["Share"] = frame["share_of_unavailable"].map(
        lambda value: f"{100 * value:.2f}%"
    )
    return frame[["Evidence category", "securities", "Share"]].rename(
        columns={"securities": "Securities"}
    )


def _cohort_comparison() -> pd.DataFrame:
    frame = pd.read_csv(
        INVESTIGATION_DIR / "missing" / "missingness_by_exchange_type.csv"
    )
    selected = frame[
        (
            frame["dimension"].eq("exchange")
            & frame["value"].isin(["NYSE", "Nasdaq", "NYSE American"])
        )
        | (
            frame["dimension"].eq("security_type")
            & frame["value"].isin(
                ["common_stock", "common_equity_class"]
            )
        )
        | (
            frame["dimension"].eq("ticker_change_status")
            & frame["value"].isin(
                ["current_SEC_ticker_match", "later_SEC_ticker_evidence"]
            )
        )
    ].copy()
    labels = {
        "common_stock": "Common stock",
        "common_equity_class": "Common-equity class",
        "current_SEC_ticker_match": "Current SEC ticker match",
        "later_SEC_ticker_evidence": "SEC-evidenced later symbol",
    }
    selected["Cohort"] = selected["value"].map(labels).fillna(
        selected["value"]
    )
    selected["dimension"] = selected["dimension"].map(
        {
            "exchange": "Exchange",
            "security_type": "Security type",
            "ticker_change_status": "Ticker status",
        }
    ).fillna(selected["dimension"])
    selected["Unavailable rate"] = selected["unavailable_rate"].map(
        lambda value: f"{100 * value:.2f}%"
    )
    return selected[
        ["dimension", "Cohort", "universe", "acquired", "unavailable", "Unavailable rate"]
    ].rename(
        columns={
            "dimension": "Dimension",
            "universe": "Universe",
            "acquired": "Acquired",
            "unavailable": "Unavailable",
        }
    )


def _identity_temporal() -> pd.DataFrame:
    frame = pd.read_csv(
        INVESTIGATION_DIR / "identity" / "temporal_coverage_by_year.csv"
    )
    return frame[
        [
            "year",
            "valid_price_securities_in_year",
            "traded_securities_in_year",
            "active_at_year_end_10_session_tolerance",
            "zero_volume_observations",
        ]
    ].rename(
        columns={
            "year": "Year",
            "valid_price_securities_in_year": "Positive price",
            "traded_securities_in_year": "Positive volume",
            "active_at_year_end_10_session_tolerance": "Active at year-end",
            "zero_volume_observations": "Zero-volume rows",
        }
    )


def _anomaly_counts() -> pd.DataFrame:
    summary = _load_json(
        INVESTIGATION_DIR / "anomalies" / "anomaly_summary.json"
    )
    rows = []
    labels = {
        "retained_extreme_market_move": "Retained extreme market move",
        "near_zero_denominator": "Near-zero denominator",
        "split_adjustment_near_event": "Split adjustment nearby",
        "same_day_split_adjustment": "Same-day split adjustment",
        "unresolved_extreme_discontinuity": "Unresolved discontinuity",
        "vendor_repaired_price_discontinuity": "Vendor-repaired price",
        "same_day_dividend_adjustment": "Same-day dividend adjustment",
    }
    for key, value in summary["extreme_return_classifications"].items():
        rows.append(
            {
                "Classification": labels.get(key, key.replace("_", " ")),
                "Rows": value,
            }
        )
    return pd.DataFrame(rows).sort_values("Rows", ascending=False)


def _corporate_action_table() -> pd.DataFrame:
    split = pd.read_csv(
        INVESTIGATION_DIR
        / "anomalies"
        / "corporate_action_split_reconciliation.csv"
    )
    split["Action"] = (
        "split "
        + split["stock_split"].map(lambda value: f"{value:.4g}")
        + "; distribution "
        + split["dividend"].map(lambda value: f"{value:,.2f}")
    )
    split["Evidence"] = split["external_verified_fact"].fillna(
        "Observed simultaneous split and distribution metadata"
    )
    split["Status"] = split["resolution_status"]
    split["final_classification"] = split["final_classification"].map(
        lambda value: str(value).replace("_", " ")
    )
    split["Status"] = split["Status"].map(
        lambda value: str(value).replace("_", " ")
    )
    selected_split = split[
        ["ticker", "event_date", "Action", "final_classification", "confidence", "Status"]
    ].rename(
        columns={
            "ticker": "Security",
            "event_date": "Date",
            "final_classification": "Classification",
        }
    )
    dividend = pd.read_csv(
        INVESTIGATION_DIR
        / "anomalies"
        / "corporate_action_dividend_reconciliation.csv"
    )
    dividend = dividend.sort_values(
        ["external_confidence", "dividend_fraction_of_previous_close"],
        ascending=[True, False],
        na_position="last",
    ).head(8)
    selected_dividend = pd.DataFrame(
        {
            "Security": dividend["ticker"],
            "Date": dividend["event_date"],
            "Action": "distribution "
            + dividend["dividend"].map(lambda value: f"{value:,.2f}"),
            "Classification": dividend["final_classification"].map(
                lambda value: str(value).replace("_", " ")
            ),
            "confidence": dividend["confidence"],
            "Status": dividend["resolution_status"].map(
                lambda value: str(value).replace("_", " ")
            ),
        }
    )
    return pd.concat(
        [selected_split, selected_dividend], ignore_index=True
    )


def _quarantine_summary() -> pd.DataFrame:
    frame = pd.read_csv(
        INVESTIGATION_DIR / "anomalies" / "analysis_quarantine_candidates.csv"
    )
    result = (
        frame.groupby(["scope", "reason"], dropna=False)
        .size()
        .rename("Rows or windows")
        .reset_index()
    )
    result["scope"] = result["scope"].map(
        lambda value: str(value).replace("_", " ")
    )
    result["reason"] = result["reason"].map(
        lambda value: str(value).replace("_", " ")
    )
    return result.rename(columns={"scope": "Scope", "reason": "Reason"}).sort_values(
        "Rows or windows", ascending=False
    )


def _status_table(
    consistency_passed: bool,
    identity: dict[str, Any],
    missing: dict[str, Any],
    anomaly: dict[str, Any],
    volume: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Issue": "P0.1 Provenance and abstract",
                "Status": "resolved" if consistency_passed else "partially_resolved",
                "Conclusion": "Yahoo is authoritative; EODHD is outside the base; abstract is generated.",
                "Remaining uncertainty": "None for provider attribution.",
            },
            {
                "Issue": "P0.2 Identity and 2025 coverage",
                "Status": "resolved",
                "Conclusion": "All checkpoints pass mapping/schema tests; no duplicate histories.",
                "Remaining uncertainty": (
                    f"{identity['stopped_trading_candidates']} trailing "
                    "zero-volume candidates are retained for review."
                ),
            },
            {
                "Issue": "P1.1 Missing histories",
                "Status": "partially_resolved",
                "Conclusion": (
                    f"{missing['unavailable']:,} failures categorized; "
                    f"{missing['safe_single_alias_candidates']:,} safe alias "
                    "candidates identified."
                ),
                "Remaining uncertainty": "Alias histories require identity continuity checks.",
            },
            {
                "Issue": "P1.2 Extreme prices and returns",
                "Status": "partially_resolved",
                "Conclusion": (
                    f"{anomaly['extreme_return_classifications']['near_zero_denominator']:,} "
                    "near-zero denominators and reverse-split levels are separated "
                    "from market moves."
                ),
                "Remaining uncertainty": (
                    f"{anomaly['extreme_return_classifications']['unresolved_extreme_discontinuity']:,} "
                    "unresolved discontinuities need independent quote evidence."
                ),
            },
            {
                "Issue": "P1.3 Volume quality",
                "Status": "partially_resolved",
                "Conclusion": (
                    f"{volume['zero_volume_rows']:,} zero-volume rows are "
                    "concentrated; split-volume normalization is not established."
                ),
                "Remaining uncertainty": "Illiquidity, suspensions, and stale rows remain observationally equivalent.",
            },
            {
                "Issue": "P1.4 Corporate actions",
                "Status": "partially_resolved",
                "Conclusion": (
                    f"{anomaly['split_flags_reconciled']:,} original split flags "
                    "are combined actions; no simple split failures remain."
                ),
                "Remaining uncertainty": "Yahoo lacks structured event-type labels for most distributions.",
            },
        ]
    )


def _readiness(summary: dict[str, Any]) -> pd.DataFrame:
    anomaly = _load_json(
        INVESTIGATION_DIR / "anomalies" / "anomaly_summary.json"
    )
    frame = pd.DataFrame(
        [
            {
                "Use": "Descriptive coverage and raw-field analysis",
                "Readiness": "ready_for_descriptive_analysis",
                "Condition": "Report the acquired subset and unavailable cohort explicitly.",
            },
            {
                "Use": "Feature engineering after chronological split",
                "Readiness": "requires_additional_validation",
                "Condition": (
                    "Resolve return eligibility, missing-history sensitivity, and "
                    "volume-convention rules before feature construction."
                ),
            },
            {
                "Use": "Return diagnostics",
                "Readiness": "quarantined_or_excluded_pending_investigation",
                "Condition": (
                    f"{anomaly['quarantine_rows']:,} rows/windows across "
                    f"{anomaly['quarantine_securities']:,} securities are flagged."
                ),
            },
            {
                "Use": "Level-based price transformations",
                "Readiness": "quarantined_or_excluded_pending_investigation",
                "Condition": f"{anomaly['extreme_price_securities']:,} securities require review.",
            },
            {
                "Use": "Large-scale targeted alias recovery",
                "Readiness": "partially_resolved",
                "Condition": (
                    f"{summary['safe_single_alias_candidates']:,} safe single-symbol "
                    "candidates remain outside the base pending admission checks."
                ),
            },
        ]
    )
    frame["Readiness"] = frame["Readiness"].map(
        lambda value: str(value).replace("_", " ")
    )
    return frame


def integrate() -> dict[str, Any]:
    identity = _load_json(
        INVESTIGATION_DIR / "identity" / "identity_findings.json"
    )
    missing = _load_json(
        INVESTIGATION_DIR / "missing" / "missing_security_summary.json"
    )
    anomaly = _load_json(
        INVESTIGATION_DIR / "anomalies" / "anomaly_summary.json"
    )
    volume = _load_json(
        INVESTIGATION_DIR / "volume" / "volume_investigation_summary.json"
    )
    consistency_path = (
        INVESTIGATION_DIR / "report_consistency" / "summary.json"
    )
    consistency = (
        _load_json(consistency_path)
        if consistency_path.exists()
        else {"passed": False}
    )

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    ISSUE_DIR.mkdir(parents=True, exist_ok=True)

    status = _status_table(
        bool(consistency.get("passed")),
        identity,
        missing,
        anomaly,
        volume,
    )
    allowed_statuses = {
        "resolved",
        "partially_resolved",
        "unresolved",
        "not_reproducible",
    }
    invalid_statuses = sorted(set(status["Status"]) - allowed_statuses)
    if invalid_statuses:
        raise ValueError(
            f"Unsupported P0/P1 report statuses: {invalid_statuses}"
        )
    _table(
        status,
        "table_p0_p1_status.tex",
        widths="p{0.16\\linewidth}p{0.18\\linewidth}p{0.27\\linewidth}p{0.27\\linewidth}",
        small=False,
    )
    _table(_failure_categories(), "table_missing_failure_categories.tex", widths="p{0.52\\linewidth}rr", small=True)
    cohort = _cohort_comparison()
    _table(cohort, "table_missing_cohort.tex", widths="p{0.17\\linewidth}p{0.35\\linewidth}rrrr", small=True)
    temporal = _identity_temporal()
    _table(temporal, "table_identity_temporal.tex", widths="rrrrr", small=True)
    anomaly_counts = _anomaly_counts()
    _table(anomaly_counts, "table_anomaly_return_classes.tex", widths="p{0.70\\linewidth}r", small=True)
    corporate = _corporate_action_table()
    _table(
        corporate,
        "table_corporate_action_reconciliation.tex",
        widths="llp{0.30\\linewidth}p{0.24\\linewidth}ll",
        small=True,
    )
    volume_year = pd.read_csv(
        INVESTIGATION_DIR / "volume" / "zero_volume_by_year.csv"
    )
    volume_year["Zero rate"] = volume_year["zero_volume_rate"].map(
        lambda value: f"{100 * value:.2f}%"
    )
    volume_display = volume_year[
        [
            "year",
            "securities",
            "zero_volume_rows",
            "Zero rate",
            "securities_zero_rate_gt_50pct",
        ]
    ].rename(
        columns={
            "year": "Year",
            "securities": "Securities",
            "zero_volume_rows": "Zero-volume rows",
            "securities_zero_rate_gt_50pct": "Securities >50 percent zero",
        }
    )
    _table(volume_display, "table_volume_investigation_by_year.tex", widths="rrrrr", small=True)
    quarantine = _quarantine_summary()
    _table(
        quarantine,
        "table_quarantine_summary.tex",
        widths="p{0.15\\linewidth}p{0.60\\linewidth}r",
        small=True,
    )
    readiness = _readiness(missing)
    _table(
        readiness,
        "table_dataset_readiness.tex",
        widths="p{0.27\\linewidth}p{0.29\\linewidth}p{0.34\\linewidth}",
        small=False,
    )

    macros = {
        "IdentityValid": identity["identity_valid"],
        "IdentityInvalid": identity["identity_invalid"],
        "PlaceholderCheckpoints": identity["placeholder_row_checkpoints"],
        "PlaceholderRows": identity["placeholder_rows"],
        "ContaminationGroups": identity["contamination_groups"],
        "StoppedTradingCandidates": identity["stopped_trading_candidates"],
        "UnavailableSecurities": missing["unavailable"],
        "AcquiredShare": f"{100 * missing['acquisition_rate']:.2f}\\%",
        "UnavailableShare": f"{100 * (1 - missing['acquisition_rate']):.2f}\\%",
        "SafeAliasCandidates": missing["safe_single_alias_candidates"],
        "AmbiguousAliasCandidates": missing["ambiguous_alias_candidates"],
        "MissingNoAlias": missing["no_later_symbol_evidence"],
        "ProbeAttempts": missing["probe_attempts"],
        "ProbeFound": missing["probe_vendor_data_found"],
        "ExtremeReturnRows": anomaly["extreme_return_rows"],
        "NearZeroReturns": anomaly["extreme_return_classifications"]["near_zero_denominator"],
        "RetainedExtremes": anomaly["extreme_return_classifications"]["retained_extreme_market_move"],
        "UnresolvedExtremes": anomaly["extreme_return_classifications"]["unresolved_extreme_discontinuity"],
        "QuarantineRows": anomaly["quarantine_rows"],
        "QuarantineSecurities": anomaly["quarantine_securities"],
        "ExtremePriceSecurities": anomaly["extreme_price_securities"],
        "SplitFlagsReconciled": anomaly["split_flags_reconciled"],
        "DividendFlagsReconciled": anomaly["dividend_flags_reconciled"],
        "ZeroVolumeRows": volume["zero_volume_rows"],
        "ZeroVolumeSecurities": volume["securities_with_zero_volume"],
        "ZeroHalfSecurities": volume["securities_for_first_half"],
        "ProlongedZeroCases": volume["prolonged_cases"],
        "SplitVolumeEvents": volume["split_events_with_volume_metrics"],
        "ConsistencyChecksPassed": consistency.get("checks_passed", 0),
        "ConsistencyChecksTotal": consistency.get("checks_total", 0),
    }
    lines = ["% Generated by src/investigation/integrate_report.py"]
    lines.extend(
        rf"\newcommand{{\{name}}}{{{value}}}"
        for name, value in macros.items()
    )
    _write(GEN_DIR / "investigation_macros.tex", "\n".join(lines))

    issue_rows = [
        {
            "priority": "P0",
            "issue": "Provider provenance and abstract consistency",
            "status": "resolved" if consistency.get("passed") else "partially_resolved",
            "severity": "critical",
            "evidence": "Yahoo manifest, raw files, provider values, and base IDs reconcile.",
            "uncertainty": "None for provider attribution.",
            "recommended_action": "Retain the generated abstract and consistency gate.",
        },
        {
            "priority": "P0",
            "issue": "Ticker identity and 2025 temporal coverage",
            "status": "resolved",
            "severity": "critical",
            "evidence": (
                f"{identity['identity_valid']:,}/{identity['checkpoints']:,} "
                "checkpoints identity-valid; no duplicate histories."
            ),
            "uncertainty": (
                f"{identity['stopped_trading_candidates']:,} trailing "
                "zero-volume candidates lack verified delisting dates."
            ),
            "recommended_action": "Retain candidates for time-series-specific review.",
        },
        {
            "priority": "P1",
            "issue": "Unavailable Yahoo histories",
            "status": "partially_resolved",
            "severity": "high",
            "evidence": f"{missing['unavailable']:,} histories categorized from manifests and SEC metadata.",
            "uncertainty": "A provider miss is not a delisting determination.",
            "recommended_action": "Run identity-checked alias recovery before any larger retry.",
        },
        {
            "priority": "P1",
            "issue": "Extreme prices and returns",
            "status": "partially_resolved",
            "severity": "high",
            "evidence": f"{anomaly['quarantine_rows']:,} diagnostic rows/windows quarantined without deleting raw data.",
            "uncertainty": "Some extreme discontinuities require independent quote evidence.",
            "recommended_action": "Exclude quarantine flags from later return diagnostics.",
        },
        {
            "priority": "P1",
            "issue": "Zero-volume concentration and split volume",
            "status": "partially_resolved",
            "severity": "medium",
            "evidence": f"{volume['zero_volume_rows']:,} zero-volume rows audited across securities and years.",
            "uncertainty": "Vendor volume normalization across splits is not documented.",
            "recommended_action": "Define future liquidity and volume-normalization rules explicitly.",
        },
        {
            "priority": "P1",
            "issue": "Corporate-action continuity",
            "status": "partially_resolved",
            "severity": "medium",
            "evidence": "All split flags reproduced as combined actions; distribution flags categorized.",
            "uncertainty": "Most distribution types lack an independently verified structured source.",
            "recommended_action": "Use action-specific return checks rather than one simple dividend rule.",
        },
    ]
    issues = pd.DataFrame(issue_rows)
    issues.to_csv(ISSUE_DIR / "p0_p1_issue_summary.csv", index=False)
    readiness.to_csv(ISSUE_DIR / "dataset_readiness.csv", index=False)

    summary_lines = [
        "# P0/P1 investigation summary",
        "",
        "This summary is generated from the identity, missing-security, anomaly, "
        "volume, and report-consistency artifacts.",
        "",
        f"- Identity-valid checkpoints: {identity['identity_valid']:,}/{identity['checkpoints']:,}.",
        f"- Unavailable Yahoo histories: {missing['unavailable']:,}.",
        f"- Safe later-symbol recovery candidates not admitted: {missing['safe_single_alias_candidates']:,}.",
        f"- Extreme return rows: {anomaly['extreme_return_rows']:,}; near-zero denominators: {anomaly['extreme_return_classifications']['near_zero_denominator']:,}.",
        f"- Quarantine rows/windows: {anomaly['quarantine_rows']:,} across {anomaly['quarantine_securities']:,} securities.",
        f"- Zero-volume rows: {volume['zero_volume_rows']:,} across {volume['securities_with_zero_volume']:,} securities.",
        f"- Report consistency checks passed: {consistency.get('checks_passed', 0):,}/{consistency.get('checks_total', 0):,}.",
        "",
        "The unresolved cases remain in the detailed CSV evidence and are not "
        "silently removed from the raw archive.",
    ]
    _write(PROJECT_ROOT / "docs" / "p0_p1_investigation_summary.md", "\n".join(summary_lines))

    changelog = """# P0/P1 remediation changelog

- Replaced the stale hardcoded abstract with `generated/abstract.tex`.
- Added a report/manifest/base consistency checker.
- Added checkpoint identity, duplicate-history, and temporal-coverage audits.
- Added missing-cohort reason, cohort-difference, and targeted-recovery analyses.
- Added independent extreme-return, extreme-level, split, and distribution reconciliation.
- Added zero-volume concentration, prolonged-case, and split-neighborhood volume diagnostics.
- Added reversible analysis-quarantine flags without deleting or rewriting raw observations.
- Regenerated report tables, figures, issue summaries, and readiness classifications.
- Corrected year formatting in generated LaTeX tables and synchronized the reconciled dividend-event count.
"""
    _write(PROJECT_ROOT / "docs" / "changelog.md", changelog)
    return {
        "status_rows": len(status),
        "issue_rows": len(issues),
        "readiness_rows": len(readiness),
        "consistency_passed": bool(consistency.get("passed")),
        "macros": macros,
    }


def main() -> None:
    print(json.dumps(integrate(), indent=2))


if __name__ == "__main__":
    main()
