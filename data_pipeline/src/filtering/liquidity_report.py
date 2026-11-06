"""Generate reproducible assets for the liquidity-filter supplement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, atomic_write_json


REPORT_DIR = PROJECT_ROOT / "reports" / "liquidity_report"
GEN_DIR = REPORT_DIR / "generated"
TABLE_SOURCE = PROJECT_ROOT / "reports" / "tables" / "liquidity"
FIGURE_SOURCE = PROJECT_ROOT / "reports" / "figures" / "liquidity"
FILTERED_ROOT = PROJECT_ROOT / "data" / "filtered" / "adtv100m_2023"
QUALITY_ROOT = FILTERED_ROOT / "quality"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _tex(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "Unknown"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}"
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


def _table(frame: pd.DataFrame, name: str, align: str | None = None) -> None:
    columns = list(frame.columns)
    align = align or ("l" + "r" * max(0, len(columns) - 1))
    lines = [
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(_tex(column.replace("_", " ")) for column in columns)
        + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_tex(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(GEN_DIR / name, "\n".join(lines))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _macro(name: str, value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        rendered = "Yes" if bool(value) else "No"
    elif isinstance(value, (int, np.integer)):
        rendered = f"{int(value):,}"
    elif isinstance(value, (float, np.floating)):
        rendered = f"{float(value):,.4f}"
    else:
        rendered = _tex(value)
    return rf"\newcommand{{\{name}}}{{{rendered}}}"


def _pct(numerator: float, denominator: float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else np.nan


def _prepare_tables(
    *,
    flow: pd.DataFrame,
    composition: pd.DataFrame,
    quantiles: pd.DataFrame,
    sensitivity: pd.DataFrame,
    before_after: pd.DataFrame,
    quality_year: pd.DataFrame,
    missing_intervals: pd.DataFrame,
    extreme_classes: pd.DataFrame,
    action_classes: pd.DataFrame,
    multi_issuers: pd.DataFrame,
    identity_candidates: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    flow_table = flow.rename(
        columns={"stage": "Selection stage", "securities": "Securities"}
    ).copy()
    flow_table["Retained from prior stage"] = [
        "100.00%",
        f"{_pct(flow.iloc[1]['securities'], flow.iloc[0]['securities']):.2f}%",
        f"{_pct(flow.iloc[2]['securities'], flow.iloc[1]['securities']):.2f}%",
        f"{_pct(flow.iloc[3]['securities'], flow.iloc[2]['securities']):.2f}%",
        "100.00%",
    ]
    _table(flow_table, "table_selection_flow.tex", "p{0.46\\linewidth}rr")

    comp = composition.copy()
    comp["dimension"] = comp["dimension"].replace(
        {"exchange": "Exchange", "security_type": "Security type"}
    )
    comp["retention_rate"] = comp["retention_rate"].map(
        lambda value: f"{100 * value:.2f}%"
    )
    comp = comp.rename(
        columns={
            "dimension": "Dimension",
            "value": "Category",
            "acquired_securities": "Acquired",
            "selected_securities": "Selected",
            "retention_rate": "Retention",
        }
    )
    _table(comp, "table_composition.tex", "llrrr")

    quant = quantiles.copy()
    quant["adtv20_usd"] = [
        f"{value:,.0f}" if statistic == "count" else f"{value / 1_000_000:,.2f}"
        for statistic, value in zip(
            quant["statistic"], quant["adtv20_usd"], strict=True
        )
    ]
    quant = quant.rename(
        columns={"statistic": "Statistic", "adtv20_usd": "ADTV20 (USD m)"}
    )
    _table(quant, "table_adtv_quantiles.tex", "lr")

    sens = sensitivity.copy()
    sens["threshold_usd"] = sens["threshold_usd"].map(
        lambda value: f"{value / 1_000_000:,.0f}"
    )
    sens["binding"] = sens["binding"].map(
        lambda value: "Yes" if bool(value) else "No"
    )
    sens = sens.rename(
        columns={
            "threshold_usd": "Threshold (USD m)",
            "securities_at_or_above": "Securities",
            "binding": "Binding",
        }
    )
    _table(sens, "table_sensitivity.tex", "rrl")

    ba = before_after.copy()
    ba["Original rate"] = ba["Original rate"].map(
        lambda value: "" if pd.isna(value) else f"{value:.3f}%"
    )
    ba["Filtered rate"] = ba["Filtered rate"].map(
        lambda value: "" if pd.isna(value) else f"{value:.3f}%"
    )
    ba = ba[
        [
            "Metric",
            "Original",
            "Filtered",
            "Original rate",
            "Filtered rate",
            "Original affected securities",
            "Filtered affected securities",
        ]
    ].rename(
        columns={
            "Original affected securities": "Original affected",
            "Filtered affected securities": "Filtered affected",
        }
    )
    _table(ba, "table_before_after.tex", "p{0.31\\linewidth}rrrrrrr")

    quality = quality_year.rename(
        columns={
            "year": "Year",
            "raw_rows": "Raw rows",
            "missing_adj_close_rows": "Missing adj. close",
            "missing_volume_rows": "Missing volume",
            "zero_volume_rows": "Zero volume",
            "securities_with_valid_observations": "Securities",
            "validated_rows": "Validated rows",
        }
    )
    _table(quality, "table_quality_by_year.tex", "r" * len(quality.columns))

    intervals = missing_intervals.copy()
    intervals = intervals.rename(
        columns={
            "ticker": "Ticker",
            "field": "Field",
            "start_date": "Start",
            "end_date": "End",
            "row_count": "Rows",
            "interval_class": "Classification",
        }
    )[
        ["Ticker", "Field", "Start", "End", "Rows", "Classification"]
    ]
    _table(intervals, "table_missing_intervals.tex", "llllrl")

    extremes = extreme_classes.rename(
        columns={"classification": "Classification", "rows": "Rows"}
    ).copy()
    _table(extremes, "table_extreme_classes.tex", "lr")

    actions = action_classes.rename(
        columns={"classification": "Classification", "events": "Case records"}
    ).copy()
    _table(actions, "table_corporate_actions.tex", "lr")

    issuers = multi_issuers.rename(
        columns={
            "cik": "SEC CIK",
            "securities": "Securities",
            "tickers": "Tickers",
            "company_names": "Company",
            "exchanges": "Exchange",
            "security_types": "Type",
        }
    )[
        ["SEC CIK", "Securities", "Tickers", "Company", "Exchange", "Type"]
    ]
    _table(issuers, "table_multi_issuers.tex", "rrp{0.22\\linewidth}llp{0.3\\linewidth}")

    identities = identity_candidates.rename(
        columns={
            "ticker_2023": "Ticker",
            "company_name": "Company",
            "quarantine_action": "Review action",
            "first_valid_date": "First valid date",
            "last_valid_date": "Last valid date",
            "n_rows": "Raw rows",
            "all_nan_price_rows": "All-NaN rows",
        }
    )
    identity_columns = [
        column
        for column in [
            "Ticker",
            "Company",
            "Review action",
            "First valid date",
            "Last valid date",
            "Raw rows",
            "All-NaN rows",
        ]
        if column in identities
    ]
    _table(
        identities[identity_columns],
        "table_identity_candidates.tex",
        "p{0.14\\linewidth}p{0.30\\linewidth}p{0.22\\linewidth}rrrr",
    )

    config_rows = pd.DataFrame(
        [
            ("Calendar", config["project"]["calendar"]),
            ("Window end", config["project"]["window_end"]),
            ("Window sessions", config["project"]["window_sessions"]),
            ("Threshold (USD)", f"{config['project']['threshold_usd']:,}"),
            ("Price basis", config["project"]["price_basis"]),
            ("Volume basis", config["project"]["volume_basis"]),
            (
                "Future split adjustment end",
                config["project"]["future_split_adjustment"]["end_date"],
            ),
        ],
        columns=["Parameter", "Value"],
    )
    _table(config_rows, "table_configuration.tex", "lp{0.65\\linewidth}")


def _save_figures(
    *,
    composition: pd.DataFrame,
    quality_year: pd.DataFrame,
    extreme_classes: pd.DataFrame,
    action_classes: pd.DataFrame,
) -> None:
    FIGURE_SOURCE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )

    selected = composition.copy()
    selected["label"] = (
        selected["dimension"].str.replace("_", " ")
        + ": "
        + selected["value"].astype(str)
    )
    selected = selected.sort_values(["dimension", "acquired_securities"])
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.barh(y - 0.18, selected["acquired_securities"], 0.34, color="#65727A", label="Acquired")
    ax.barh(y + 0.18, selected["selected_securities"], 0.34, color="#4E8C86", label="Selected")
    ax.set_yticks(y, selected["label"])
    ax.set_xlabel("Securities")
    ax.set_title("Acquired and selected cohort composition")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIGURE_SOURCE / "liquidity_composition.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(
        quality_year["year"],
        quality_year["missing_adj_close_rows"],
        marker="o",
        color="#19324D",
        label="Missing adjusted close",
    )
    ax.plot(
        quality_year["year"],
        quality_year["zero_volume_rows"],
        marker="s",
        color="#B28A48",
        label="Zero volume",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Rows")
    ax.set_title("Filtered-cohort missingness and zero-volume rows by year")
    ax.legend()
    ax.grid(alpha=0.35)
    fig.savefig(FIGURE_SOURCE / "liquidity_quality_by_year.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = (
        extreme_classes["classification"].str.replace("_", " ")
        if not extreme_classes.empty
        else pd.Series(dtype=str)
    )
    values = extreme_classes["rows"] if not extreme_classes.empty else pd.Series(dtype=int)
    if not values.empty:
        ax.barh(labels, values, color="#3F6F8F")
        ax.set_xlabel("Diagnostic rows")
    ax.set_title("Extreme diagnostic-return classifications")
    fig.savefig(FIGURE_SOURCE / "liquidity_extreme_classes.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    labels = (
        action_classes["classification"].str.replace("_", " ")
        if not action_classes.empty
        else pd.Series(dtype=str)
    )
    values = action_classes["events"] if not action_classes.empty else pd.Series(dtype=int)
    if not values.empty:
        ax.barh(labels, values, color="#A45D4F")
        ax.set_xlabel("Case records")
    ax.set_title("Corporate-action cases in the filtered cohort")
    fig.savefig(FIGURE_SOURCE / "liquidity_corporate_action_cases.pdf")
    plt.close(fig)


def build(config_path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metadata = _load_json(FILTERED_ROOT / "filter_metadata.json")
    integrity = _load_json(QUALITY_ROOT / "post_filter_integrity_summary.json")
    anomalies = _load_json(QUALITY_ROOT / "post_filter_anomaly_summary.json")

    selected_master = pd.read_parquet(
        FILTERED_ROOT / "selected_security_master.parquet"
    )
    selected_daily = pd.read_parquet(
        FILTERED_ROOT / "selected_equity_daily.parquet"
    )
    screen_all = pd.read_csv(FILTERED_ROOT / "liquidity_screen_all.csv")
    original_base = pd.read_parquet(
        PROJECT_ROOT / "data/validated/base.parquet",
        columns=["security_id"],
    )
    flow = pd.read_csv(TABLE_SOURCE / "liquidity_selection_flow.csv")
    composition = pd.read_csv(TABLE_SOURCE / "liquidity_composition.csv")
    quantiles = pd.read_csv(TABLE_SOURCE / "liquidity_adtv_quantiles.csv")
    sensitivity = pd.read_csv(TABLE_SOURCE / "liquidity_sensitivity.csv")
    before_after = pd.read_csv(QUALITY_ROOT / "before_after_metrics.csv")
    quality_year = pd.read_csv(QUALITY_ROOT / "quality_by_year.csv")
    missing_intervals = pd.read_csv(QUALITY_ROOT / "missing_value_intervals.csv")
    extreme_classes = pd.read_csv(
        QUALITY_ROOT / "filtered_extreme_return_classes.csv"
    )
    action_classes = pd.read_csv(TABLE_SOURCE / "postfilter_corporate_action_classes.csv")
    multi_issuers = pd.read_csv(QUALITY_ROOT / "multi_security_issuers.csv")
    identity_candidates = pd.read_csv(
        QUALITY_ROOT / "identity_review_candidates.csv"
    )

    selected_ids = set(selected_master["security_id"].astype(str))
    daily_ids = set(selected_daily["security_id"].astype(str))
    if selected_ids != daily_ids:
        raise ValueError("Selected master and daily IDs do not reconcile")
    if selected_daily.duplicated(["security_id", "date"]).any():
        raise ValueError("Selected daily data contain duplicate security-date keys")
    if not selected_daily["pct_return"].isna().all():
        raise ValueError("The research-base pct_return contract must remain null")
    if not selected_daily["date"].between("2016-01-01", "2025-12-31").all():
        raise ValueError("Selected daily data fall outside 2016-2025")

    selected_screen = screen_all.loc[screen_all["selected"].fillna(False)].copy()
    if set(selected_screen["security_id"].astype(str)) != selected_ids:
        raise ValueError("Screen-selected IDs differ from selected daily IDs")
    if not selected_screen["adtv20_usd"].ge(metadata["threshold_usd"]).all():
        raise ValueError("A selected security is below the inclusive threshold")

    recommended = {
        "selected_securities": int(len(selected_master)),
        "selected_distinct_issuers": int(selected_master["cik"].nunique()),
        "validated_rows": int(len(selected_daily)),
        "universe_securities": int(metadata["universe_securities"]),
        "acquired_securities": int(metadata["acquired_securities"]),
        "complete_window_securities": int(metadata["complete_window_securities"]),
        "below_threshold": int(
            screen_all["screen_status"].eq("excluded_below_threshold").sum()
        ),
        "insufficient_window": int(
            screen_all["screen_status"].eq("insufficient_liquidity_window_data").sum()
        ),
        "not_acquired": int(
            screen_all["screen_status"].eq("not_acquired").sum()
        ),
        "unresolved_price_volume": int(
            screen_all["screen_status"].eq("unresolved_price_volume_issue").sum()
        ),
    }
    for key, value in recommended.items():
        if key == "validated_rows":
            expected = integrity["validated_rows"]
        elif key == "selected_securities":
            expected = integrity["selected_securities"]
        elif key == "selected_distinct_issuers":
            expected = integrity["selected_distinct_issuers"]
        elif key == "universe_securities":
            expected = metadata["universe_securities"]
        elif key == "acquired_securities":
            expected = metadata["acquired_securities"]
        elif key == "complete_window_securities":
            expected = metadata["complete_window_securities"]
        else:
            expected = recommended[key]
        if int(expected) != value:
            raise ValueError(f"Snapshot inconsistency for {key}: {value} vs {expected}")

    adtv = selected_screen["adtv20_usd"]
    selected_with_future_splits = int(
        selected_screen["future_split_events_restored"].gt(0).sum()
    )
    future_split_events = int(
        selected_screen["future_split_events_restored"].sum()
    )
    window_sessions = metadata["window_sessions"]
    if len(window_sessions) != int(metadata["window_session_count"]):
        raise ValueError("Window session list does not match its count")
    if window_sessions[-1] != "2022-12-30" or window_sessions[0] != "2022-12-02":
        raise ValueError("Unexpected final-twenty-session window")

    macros = [
        "% Generated by src/filtering/liquidity_report.py; do not edit by hand.",
        _macro("UniverseSecurities", recommended["universe_securities"]),
        _macro("OriginalValidatedObservations", len(original_base)),
        _macro("OriginalDiagnosticRows", anomalies["original_diagnostic_metrics"]["rows"]),
        _macro("FilteredDiagnosticRows", anomalies["filtered_diagnostic_metrics"]["rows"]),
        _macro("AcquiredSecurities", recommended["acquired_securities"]),
        _macro("CompleteWindowSecurities", recommended["complete_window_securities"]),
        _macro("SelectedSecurities", recommended["selected_securities"]),
        _macro("SelectedIssuers", recommended["selected_distinct_issuers"]),
        _macro("SelectedObservations", recommended["validated_rows"]),
        _macro("BelowThreshold", recommended["below_threshold"]),
        _macro("InsufficientWindow", recommended["insufficient_window"]),
        _macro("NotAcquired", recommended["not_acquired"]),
        _macro("UnresolvedPriceVolume", recommended["unresolved_price_volume"]),
        _macro("AcquiredRetentionPct", 100 * metadata["acquired_retention_rate"]),
        _macro("ObservationRetentionPct", 100 * metadata["observation_retention_rate"]),
        _macro("LiquidityThreshold", f"USD {metadata['threshold_usd']:,.0f}"),
        _macro("LiquidityWindowStart", window_sessions[0]),
        _macro("LiquidityWindowEnd", window_sessions[-1]),
        _macro("LiquidityWindowSessions", len(window_sessions)),
        _macro("SelectedADTVMinimum", f"USD {adtv.min():,.0f}"),
        _macro("SelectedADTVMedian", f"USD {adtv.median():,.0f}"),
        _macro("SelectedADTVMaximum", f"USD {adtv.max():,.0f}"),
        _macro("SelectedWithFutureSplitAdjustment", selected_with_future_splits),
        _macro("FutureSplitEventsRestored", future_split_events),
        _macro("RawSelectedRows", integrity["raw_rows"]),
        _macro("PlaceholderRows", integrity["placeholder_all_nan_rows"]),
        _macro("MissingAdjClose", integrity["missing_adj_close_rows"]),
        _macro("MissingVolume", integrity["missing_volume_rows"]),
        _macro("MissingValueIntervals", integrity["missing_value_intervals"]),
        _macro("IncompleteHistorySecurities", integrity["incomplete_history_securities"]),
        _macro("InternalCalendarGaps", integrity["internal_calendar_gaps"]),
        _macro("ZeroVolumeRows", integrity["zero_volume_rows"]),
        _macro("ZeroVolumeSecurities", anomalies["zero_volume_securities"]),
        _macro("ProlongedZeroVolumeSecurities", anomalies["prolonged_zero_volume_securities"]),
        _macro("ExtremeReturns", integrity["extreme_return_rows"]),
        _macro("ExtremeReturnSecurities", anomalies["filtered_diagnostic_metrics"]["extreme_return_securities"]),
        _macro(
            "SameDayDividendAdjustmentCases",
            anomalies["extreme_return_classifications"].get(
                "same_day_dividend_adjustment", 0
            ),
        ),
        _macro(
            "RetainedExtremeMarketMoves",
            anomalies["extreme_return_classifications"].get(
                "retained_extreme_market_move", 0
            ),
        ),
        _macro("UnresolvedExtremeCases", integrity["unresolved_extreme_cases"]),
        _macro("ExtremeVolumeSpikes", anomalies["filtered_diagnostic_metrics"]["extreme_volume_spikes"]),
        _macro("ExtremeVolumeSpikeSecurities", anomalies["filtered_diagnostic_metrics"]["extreme_volume_spike_securities"]),
        _macro("SplitEvents", anomalies["corporate_actions"]["split_events"]),
        _macro("DividendEvents", anomalies["corporate_actions"]["dividend_events"]),
        _macro("CorporateActionCaseRecords", int(action_classes["events"].sum())),
        _macro("MultiIssuerCount", len(multi_issuers)),
        _macro("IdentityReviewCandidates", integrity["identity_review_candidates"]),
        _macro("QuarantineRowsWindows", anomalies["quarantine_rows_windows"]),
        _macro("ProtectedInputsUnchanged", integrity["protected_inputs_unchanged"]),
    ]
    _write(GEN_DIR / "macros.tex", "\n".join(macros))

    _prepare_tables(
        flow=flow,
        composition=composition,
        quantiles=quantiles,
        sensitivity=sensitivity,
        before_after=before_after,
        quality_year=quality_year,
        missing_intervals=missing_intervals,
        extreme_classes=extreme_classes,
        action_classes=action_classes,
        multi_issuers=multi_issuers,
        identity_candidates=identity_candidates,
        config=config,
    )
    _save_figures(
        composition=composition,
        quality_year=quality_year,
        extreme_classes=extreme_classes,
        action_classes=action_classes,
    )

    stats = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "snapshot": {
            **recommended,
            "window_sessions": window_sessions,
            "threshold_usd_inclusive": metadata["threshold_usd"],
            "selected_adtv20_usd": {
                "minimum": float(adtv.min()),
                "median": float(adtv.median()),
                "maximum": float(adtv.max()),
            },
        },
        "integrity": {
            key: integrity[key]
            for key in [
                "raw_rows",
                "placeholder_all_nan_rows",
                "missing_adj_close_rows",
                "missing_volume_rows",
                "missing_value_intervals",
                "incomplete_history_securities",
                "internal_calendar_gaps",
                "zero_volume_rows",
                "extreme_return_rows",
                "unresolved_extreme_cases",
                "multi_security_issuers",
                "identity_review_candidates",
                "identity_review_candidate_tickers",
                "protected_inputs_unchanged",
            ]
        },
        "anomalies": {
            "zero_volume_securities": anomalies["zero_volume_securities"],
            "prolonged_zero_volume_securities": anomalies[
                "prolonged_zero_volume_securities"
            ],
            "extreme_return_securities": anomalies[
                "filtered_diagnostic_metrics"
            ]["extreme_return_securities"],
            "extreme_volume_spikes": anomalies["filtered_diagnostic_metrics"][
                "extreme_volume_spikes"
            ],
            "extreme_volume_spike_securities": anomalies[
                "filtered_diagnostic_metrics"
            ]["extreme_volume_spike_securities"],
            "quarantine_rows_windows": anomalies["quarantine_rows_windows"],
            "corporate_actions": anomalies["corporate_actions"],
        },
    }
    atomic_write_json(stats, GEN_DIR / "report_statistics.json")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/liquidity_filter.yaml"
    )
    args = parser.parse_args()
    print(json.dumps(build(args.config), indent=2))


if __name__ == "__main__":
    main()
