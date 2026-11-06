"""Generate reproducible EDA figures, tables, and report statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, atomic_write_json, configure_logging, project_path


FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
GEN_DIR = PROJECT_ROOT / "reports" / "latex" / "generated"
STATS_PATH = GEN_DIR / "report_statistics.json"

COLORS = {
    "navy": "#19324D",
    "blue": "#3F6F8F",
    "teal": "#4E8C86",
    "gold": "#B28A48",
    "rust": "#A45D4F",
    "slate": "#65727A",
    "light": "#DCE4E8",
}


def _number(value: Any, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "Not estimated"
    if decimals:
        return f"{float(value):,.{decimals}f}"
    return f"{int(value):,}"


def _pct(value: Any, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "Unknown"
    return f"{100 * float(value):.{decimals}f}%"


def _tex(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        text = str(integer) if 1900 <= integer <= 2100 else f"{integer:,}"
    elif (
        isinstance(value, (float, np.floating))
        and math.isfinite(float(value))
        and float(value).is_integer()
    ):
        integer = int(value)
        text = str(integer) if 1900 <= integer <= 2100 else f"{integer:,}"
    else:
        text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _save_csv(frame: pd.DataFrame, name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / name, index=False)


def _latex_table(
    frame: pd.DataFrame,
    path: Path,
    columns: list[str],
    headers: list[str],
    formats: list[str] | None = None,
) -> None:
    formats = formats or ["l"] * len(columns)
    spec = "".join(formats)
    lines = [rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule"]
    lines.append(" & ".join(_tex(v) for v in headers) + r" \\")
    lines.append(r"\midrule")
    for values in frame[columns].itertuples(index=False, name=None):
        lines.append(
            " & ".join(_tex(values[position]) for position in range(len(columns)))
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(path, "\n".join(lines))


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#6D7A80",
            "axes.linewidth": 0.7,
            "grid.color": "#DCE4E8",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, format="pdf")
    plt.close(fig)


def _load_inputs() -> dict[str, Any]:
    return {
        "universe": pd.read_csv(PROJECT_ROOT / "data/universe/universe_2023.csv", low_memory=False),
        "master": pd.read_parquet(PROJECT_ROOT / "data/universe/security_master.parquet"),
        "universe_issues": pd.read_csv(PROJECT_ROOT / "data/universe/universe_issues.csv", low_memory=False),
        "yahoo_manifest": pd.read_csv(PROJECT_ROOT / "data/metadata/download_manifest.csv", low_memory=False),
        "yahoo_failures": pd.read_csv(PROJECT_ROOT / "data/metadata/download_failures.csv", low_memory=False),
        "demo_manifest": pd.read_csv(PROJECT_ROOT / "data/metadata/eodhd_demo_manifest.csv", low_memory=False),
        "demo_failures": pd.read_csv(PROJECT_ROOT / "data/metadata/eodhd_demo_failures.csv", low_memory=False),
        "base": pd.read_parquet(PROJECT_ROOT / "data/validated/base.parquet"),
        "diagnostics": pd.read_parquet(PROJECT_ROOT / "data/validated/diagnostic_returns.parquet"),
        "quality_security": pd.read_csv(PROJECT_ROOT / "data/validated/data_quality_by_security.csv", low_memory=False),
        "quality_year": pd.read_csv(PROJECT_ROOT / "data/validated/data_quality_by_year.csv", low_memory=False),
        "data_issues": pd.read_csv(PROJECT_ROOT / "data/validated/data_issues.csv", low_memory=False),
        "demo_provenance": json.loads((PROJECT_ROOT / "data/metadata/eodhd_demo_provenance.json").read_text()),
        "universe_provenance": json.loads((PROJECT_ROOT / "data/metadata/universe_source_provenance.json").read_text()),
        "validation_provenance": json.loads((PROJECT_ROOT / "data/validated/validation_provenance.json").read_text()),
    }


def _compute_statistics(data: dict[str, Any]) -> dict[str, Any]:
    universe: pd.DataFrame = data["universe"]
    master: pd.DataFrame = data["master"]
    base: pd.DataFrame = data["base"]
    diag: pd.DataFrame = data["diagnostics"]
    quality_year: pd.DataFrame = data["quality_year"]
    quality_security: pd.DataFrame = data["quality_security"]
    yahoo: pd.DataFrame = data["yahoo_manifest"]
    demo: pd.DataFrame = data["demo_manifest"]

    actual_ids = set(base["security_id"].dropna().astype(str))
    successful_quality = quality_security[quality_security["security_id"].isin(actual_ids)]
    diagnostic_values = pd.to_numeric(diag["diagnostic_pct_return"], errors="coerce").dropna()
    price_values = pd.to_numeric(base["adj_close"], errors="coerce").dropna()
    volume_values = pd.to_numeric(base["volume"], errors="coerce").dropna()
    positive_volume_values = volume_values.loc[volume_values.gt(0)]

    for frame in (base, diag, quality_year, quality_security):
        for column in frame.columns:
            if column.endswith("_flag"):
                frame[column] = frame[column].fillna(False).astype(bool)

    action_by_year = (
        base.assign(year=pd.to_datetime(base["date"]).dt.year)
        .groupby("year")
        .size()
        .rename("observations")
        .reset_index()
    )
    action_by_year["split_events"] = action_by_year["year"].map(
        quality_year.set_index("year")["split_events"]
    ).fillna(0).astype(int)
    action_by_year["dividend_events"] = action_by_year["year"].map(
        quality_year.set_index("year")["dividend_events"]
    ).fillna(0).astype(int)

    split_rows = successful_quality[successful_quality["split_events"] > 0]
    split_audit: list[dict[str, Any]] = []
    dividend_audit: list[dict[str, Any]] = []
    raw_frames: dict[str, pd.DataFrame] = {}
    for _, security in split_rows.iterrows():
        security_diag = diag[diag["security_id"].eq(security["security_id"])].sort_values("date").copy()
        raw = data["quality_security"]
        del raw  # explicitly avoid using the summary table as an event source
        raw_frame = pd.read_parquet(security["raw_file"])
        raw_frame = raw_frame.sort_values("date").copy()
        raw_frames[str(security["security_id"])] = raw_frame
        raw_frame["adj_gross"] = raw_frame["adj_close"] / raw_frame["adj_close"].shift(1)
        raw_frame["raw_ratio"] = raw_frame["close"].shift(1) / raw_frame["close"]
        raw_frame["volume_ratio"] = raw_frame["volume"] / raw_frame["volume"].shift(1)
        provider = str(raw_frame.get("provider", pd.Series([""])).iloc[0]).lower()
        for position, event in raw_frame[
            raw_frame["stock_splits"].gt(0)
        ].iterrows():
            if position == 0:
                # No prior close is available to test discontinuity.
                continue
            implied = event["raw_ratio"] * event["adj_gross"]
            expected = 1.0 if "yahoo" in provider else float(event["stock_splits"])
            rel_error = abs(implied - expected) / abs(expected)
            split_audit.append(
                {
                    "ticker": security["ticker"],
                    "event_date": pd.Timestamp(event["date"]).date().isoformat(),
                    "split_factor": float(event["stock_splits"]),
                    "adjusted_return": float(event["adj_gross"] - 1),
                    "raw_diagnostic_return": float(
                        event["close"] / raw_frame.loc[position - 1, "close"] - 1
                    ),
                    "implied_factor": float(implied),
                    "relative_error": float(rel_error),
                    "volume_ratio": float(event["volume_ratio"]),
                    "status": "pass" if rel_error <= 0.03 else "review",
                }
            )
    split_audit_frame = pd.DataFrame(split_audit)
    if not split_audit_frame.empty:
        split_audit_frame = split_audit_frame.sort_values(
            ["event_date", "ticker"], kind="stable"
        ).reset_index(drop=True)
        review = split_audit_frame.sort_values(
            "relative_error", ascending=False
        ).head(6)
        passing = split_audit_frame[
            split_audit_frame["relative_error"].le(0.03)
        ]
        if not passing.empty:
            positions = np.unique(
                np.linspace(0, len(passing) - 1, min(6, len(passing)), dtype=int)
            )
            passing = passing.iloc[positions]
        split_audit_frame = (
            pd.concat([review, passing], ignore_index=True)
            .drop_duplicates(["ticker", "event_date"])
            .sort_values(["event_date", "ticker"])
            .head(12)
            .reset_index(drop=True)
        )

    all_success_ids = successful_quality["security_id"].astype(str).tolist()
    for security_id in all_success_ids:
        raw_frame = raw_frames.get(security_id)
        if raw_frame is None:
            path = successful_quality.loc[
                successful_quality["security_id"].astype(str).eq(security_id),
                "raw_file",
            ].iloc[0]
            raw_frame = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
        raw_frame = raw_frame.sort_values("date").reset_index(drop=True)
        raw_frame["adj_gross"] = raw_frame["adj_close"] / raw_frame["adj_close"].shift(1)
        raw_frame["raw_total_return"] = (
            (raw_frame["close"] + raw_frame["dividends"])
            / raw_frame["close"].shift(1)
            - 1
        )
        ticker = str(
            successful_quality.loc[
                successful_quality["security_id"].astype(str).eq(security_id),
                "ticker",
            ].iloc[0]
        )
        for _, event in raw_frame[raw_frame["dividends"].gt(0)].iterrows():
            adjusted_return = float(event["adj_gross"] - 1)
            raw_total_return = float(event["raw_total_return"])
            error = abs(adjusted_return - raw_total_return)
            dividend_audit.append(
                {
                    "ticker": ticker,
                    "event_date": pd.Timestamp(event["date"]).date().isoformat(),
                    "cash_dividend": float(event["dividends"]),
                    "adjusted_return": adjusted_return,
                    "raw_total_return": raw_total_return,
                    "absolute_return_error": error,
                    "status": "pass" if error <= 0.01 else "review",
                }
            )
    dividend_audit_frame = pd.DataFrame(dividend_audit).sort_values(
        ["event_date", "ticker"]
    )
    if not dividend_audit_frame.empty:
        review = dividend_audit_frame.sort_values(
            "absolute_return_error", ascending=False
        ).head(6)
        passing = dividend_audit_frame[
            dividend_audit_frame["absolute_return_error"].le(0.01)
        ].copy()
        passing["year"] = pd.to_datetime(passing["event_date"]).dt.year
        by_year = (
            passing.groupby("year", group_keys=False)
            .head(2)
            .sort_values(["event_date", "ticker"])
            .head(6)
        )
        dividend_audit_frame = (
            pd.concat([review, by_year], ignore_index=True)
            .drop_duplicates(["ticker", "event_date"])
            .sort_values(["event_date", "ticker"])
            .head(12)
            .reset_index(drop=True)
        )

    ext = diag[diag["extreme_return_flag"].fillna(False)].copy()
    ext = ext.reindex(ext["diagnostic_pct_return"].abs().sort_values(ascending=False).index)
    extreme_table = ext[
        ["security_id", "ticker", "date", "diagnostic_pct_return", "adj_close"]
    ].head(12)

    issue_frames = []
    for scope, frame in (("universe", data["universe_issues"]), ("validated", data["data_issues"])):
        if frame.empty:
            continue
        value_col = "severity"
        type_col = "issue_type"
        issue_frames.append(
            frame.assign(scope=scope)
            .groupby(["scope", value_col, type_col], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
    issue_summary = (
        pd.concat(issue_frames, ignore_index=True)
        if issue_frames
        else pd.DataFrame(columns=["severity", "issue_type", "count", "scope"])
    )

    exchange = (
        master.groupby("exchange").size().rename("securities").reset_index()
    )
    exchange["audit_rows"] = exchange["exchange"].map(universe["exchange"].value_counts()).fillna(0).astype(int)
    sec_type = master.groupby("security_type").size().rename("securities").reset_index()

    coverage = quality_year.copy()
    coverage["active_with_data"] = coverage["securities_with_valid_adj_close"]
    coverage["unknown_or_unavailable"] = len(master) - coverage["active_with_data"]

    missingness = quality_year[
        [
            "year",
            "observations",
            "missing_adj_close_observations",
            "missing_volume_observations",
            "expected_sessions_in_observed_intervals",
            "unobserved_sessions_within_security_intervals",
        ]
    ].copy()

    volume_flags = pd.DataFrame(
        [
            {
                "flag": "missing volume",
                "observations": int(
                    pd.to_numeric(
                        successful_quality["missing_volume_observations"],
                        errors="coerce",
                    ).fillna(0).sum()
                ),
            },
            {"flag": "negative volume", "observations": int((base["volume"] < 0).sum())},
            {"flag": "zero volume", "observations": int((base["volume"] == 0).sum())},
            {
                "flag": "extreme volume spike",
                "observations": int(diag["extreme_volume_spike_flag"].fillna(False).sum()),
            },
        ]
    )

    gap_examples = quality_security[
        quality_security["max_unobserved_gap_sessions"].fillna(0).gt(0)
    ][
        [
            "security_id",
            "ticker",
            "max_unobserved_gap_sessions",
            "history_status",
            "active_period_verification",
        ]
    ].head(20)

    auditability = {
        "unique_security_ids": int(master["security_id"].nunique()),
        "unique_tickers": int(master["ticker_2023"].nunique()),
        "missing_security_ids": int(master["security_id"].isna().sum()),
        "missing_tickers": int(master["ticker_2023"].isna().sum()),
        "duplicate_base_keys": int(base.duplicated(["security_id", "date"]).sum()),
        "pct_return_all_null": bool(base["pct_return"].isna().all()),
        "diagnostic_non_null": int(diagnostic_values.size),
    }

    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "scope": {
            "observation_start": str(pd.to_datetime(base["date"]).min().date()),
            "observation_end": str(pd.to_datetime(base["date"]).max().date()),
            "primary_provider": "Yahoo Finance via yfinance",
            "secondary_sample_provider": "EODHD public demo",
            "full_historical_coverage_established": False,
        },
        "universe": {
            "audit_rows": int(len(universe)),
            "included_securities": int(len(master)),
            "excluded_audit_rows": int((~universe["included"].astype(bool)).sum()),
            "issue_groups": int(len(data["universe_issues"])),
            "current_ticker_match": int(master["current_ticker"].fillna("").astype(str).ne("").sum()),
            "no_current_ticker_match": int(master["current_ticker"].fillna("").astype(str).eq("").sum()),
            "recorded_ticker_changes": int(master["subsequent_ticker_changes"].fillna("").astype(str).ne("").sum()),
            "reit_common_stock": int(master["security_type"].eq("reit_common_stock").sum()),
            "ordinary_shares": int(master["security_type"].eq("ordinary_shares").sum()),
            "common_equity_class": int(master["security_type"].eq("common_equity_class").sum()),
            "adr_rows": int(universe["security_type"].eq("adr").sum()),
            "fund_rows": int(universe["security_type"].eq("fund").sum()),
            "preferred_rows": int(universe["security_type"].eq("preferred").sum()),
            "exchange_counts": exchange.set_index("exchange")["securities"].to_dict(),
            "type_counts": sec_type.set_index("security_type")["securities"].to_dict(),
            "issue_severity_counts": data["universe_issues"]["severity"].value_counts().to_dict(),
            "issue_type_counts": data["universe_issues"]["issue_type"].value_counts().head(12).to_dict(),
        },
        "acquisition": {
            "yahoo_attempted": int(len(yahoo)),
            "yahoo_success": int(yahoo["download_status"].eq("success").sum()),
            "yahoo_failed": int(yahoo["download_status"].eq("failed").sum()),
            "yahoo_partial": int(yahoo["download_status"].eq("partial").sum()),
            "demo_selected": int((data["demo_provenance"].get("sample_selected") or len(demo))),
            "demo_success": int(demo["download_status"].eq("success").sum()),
            "demo_failed": int(demo["download_status"].eq("failed").sum()),
            "demo_success_tickers": sorted(demo.loc[demo["download_status"].eq("success"), "ticker_2023"].astype(str)),
            "secondary_role": data["demo_provenance"].get("empirical_role"),
        },
        "data": {
            "securities": int(base["security_id"].nunique()),
            "universe_without_observations": int(
                len(master) - base["security_id"].nunique()
            ),
            "observations": int(len(base)),
            "valid_sessions_per_security_min": int(
                successful_quality["valid_adj_close_sessions"].min()
            ),
            "valid_sessions_per_security_max": int(successful_quality["valid_adj_close_sessions"].max()),
            "expected_sessions": int(quality_year["expected_sessions_in_observed_intervals"].sum()),
            "missing_adj_close": int(
                pd.to_numeric(
                    successful_quality["missing_adj_close_observations"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "missing_volume": int(
                pd.to_numeric(
                    successful_quality["missing_volume_observations"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "nonpositive_adj_close": int(
                pd.to_numeric(
                    successful_quality["nonpositive_adj_close_observations"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "negative_volume": int(
                pd.to_numeric(
                    successful_quality["negative_volume_observations"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "zero_volume": int((base["volume"] == 0).sum()),
            "row_missing_adj_rate": float(
                int(
                    pd.to_numeric(
                        successful_quality["missing_adj_close_observations"],
                        errors="coerce",
                    ).fillna(0).sum()
                )
                / max(
                    1,
                    int(
                        pd.to_numeric(
                            successful_quality["valid_adj_close_sessions"],
                            errors="coerce",
                        ).fillna(0).sum()
                    )
                    + int(
                        pd.to_numeric(
                            successful_quality["missing_adj_close_observations"],
                            errors="coerce",
                        ).fillna(0).sum()
                    ),
                )
            ),
            "row_missing_volume_rate": float(
                int(
                    pd.to_numeric(
                        successful_quality["missing_volume_observations"],
                        errors="coerce",
                    ).fillna(0).sum()
                )
                / max(
                    1,
                    int(
                        pd.to_numeric(
                            successful_quality["valid_volume_sessions"],
                            errors="coerce",
                        ).fillna(0).sum()
                    )
                    + int(
                        pd.to_numeric(
                            successful_quality["missing_volume_observations"],
                            errors="coerce",
                        ).fillna(0).sum()
                    ),
                )
            ),
            "unobserved_sessions": int(quality_year["unobserved_sessions_within_security_intervals"].sum()),
            "price_min": float(price_values.min()),
            "price_median": float(price_values.median()),
            "price_max": float(price_values.max()),
            "volume_min": float(positive_volume_values.min()),
            "volume_median": float(positive_volume_values.median()),
            "volume_max": float(positive_volume_values.max()),
            "split_events": int(quality_year["split_events"].sum()),
            "dividend_events": int(quality_year["dividend_events"].sum()),
            "split_continuity_flags": int(
                pd.to_numeric(
                    successful_quality["split_continuity_flags"], errors="coerce"
                ).fillna(0).sum()
            ),
            "dividend_continuity_flags": int(
                pd.to_numeric(
                    successful_quality["dividend_continuity_flags"], errors="coerce"
                ).fillna(0).sum()
            ),
            "extreme_return_rows": int(diag["extreme_return_flag"].fillna(False).sum()),
            "extreme_volume_spike_rows": int(diag["extreme_volume_spike_flag"].fillna(False).sum()),
            "raw_adjusted_conflict_rows": int(diag["raw_adjusted_return_conflict_flag"].fillna(False).sum()),
            "diagnostic_min": float(diagnostic_values.min()),
            "diagnostic_max": float(diagnostic_values.max()),
        },
        "auditability": auditability,
        "issues": {
            "total_groups": int(len(issue_summary)),
            "severity_counts": issue_summary.groupby("severity")["count"].sum().to_dict(),
            "data_issue_groups": int(len(data["data_issues"])),
        },
        "tables": {
            "universe_by_exchange": exchange.to_dict("records"),
            "universe_by_type": sec_type.to_dict("records"),
            "coverage_by_year": coverage.to_dict("records"),
            "corporate_actions_by_year": action_by_year.to_dict("records"),
            "split_audit": split_audit_frame.to_dict("records"),
            "dividend_audit": dividend_audit_frame.to_dict("records"),
            "volume_flags": volume_flags.to_dict("records"),
            "issue_summary": issue_summary.to_dict("records"),
        },
    }


def _write_tables(stats: dict[str, Any], data: dict[str, Any]) -> None:
    universe_stats = stats["universe"]
    exchange = pd.DataFrame(universe_stats["exchange_counts"].items(), columns=["Exchange", "Securities"])
    exchange["Share"] = exchange["Securities"] / exchange["Securities"].sum()
    exchange["Share"] = exchange["Share"].map(lambda x: _pct(x))
    _save_csv(exchange, "universe_by_exchange.csv")
    _latex_table(
        exchange,
        GEN_DIR / "table_universe_by_exchange.tex",
        ["Exchange", "Securities", "Share"],
        ["Exchange", "Verified securities", "Share"],
        ["l", "r", "r"],
    )

    security_type = pd.DataFrame(universe_stats["type_counts"].items(), columns=["Security type", "Securities"])
    security_type["Definition"] = security_type["Security type"].map(
        {
            "common_stock": "ordinary common stock",
            "common_equity_class": "common equity class",
            "ordinary_shares": "non-ADR ordinary shares",
            "reit_common_stock": "REIT common equity",
        }
    )
    security_type["Category"] = security_type["Security type"].map(
        {
            "common_stock": "Common stock",
            "common_equity_class": "Common equity class",
            "ordinary_shares": "Ordinary shares (non-ADR)",
            "reit_common_stock": "REIT common equity",
        }
    )
    _save_csv(security_type, "universe_by_type.csv")
    _latex_table(
        security_type,
        GEN_DIR / "table_universe_by_type.tex",
        ["Category", "Definition", "Securities"],
        ["Category", "Documented definition", "Securities"],
        ["l", "l", "r"],
    )

    acq = stats["acquisition"]
    download = pd.DataFrame(
        [
            {"Provider": "Yahoo Finance (primary)", "Requested": acq["yahoo_attempted"], "Success": acq["yahoo_success"], "Failed": acq["yahoo_failed"], "Partial": acq["yahoo_partial"]},
            {"Provider": "EODHD public demo (secondary)", "Requested": acq["demo_selected"], "Success": acq["demo_success"], "Failed": acq["demo_failed"], "Partial": 0},
        ]
    )
    _save_csv(download, "download_status.csv")
    _latex_table(
        download,
        GEN_DIR / "table_download_status.tex",
        ["Provider", "Requested", "Success", "Failed", "Partial"],
        ["Provider", "Requested", "Success", "Failed", "Partial"],
        ["l", "r", "r", "r", "r"],
    )

    coverage = pd.DataFrame(stats["tables"]["coverage_by_year"])
    _save_csv(coverage, "coverage_by_year.csv")
    coverage_display = coverage.copy()
    for column in (
        "year",
        "securities_with_valid_adj_close",
        "observations",
        "active_with_data",
        "unknown_or_unavailable",
    ):
        coverage_display[column] = coverage_display[column].fillna(0).astype(int)
    _latex_table(
        coverage_display,
        GEN_DIR / "table_coverage_by_year.tex",
        ["year", "securities_with_valid_adj_close", "observations", "active_with_data", "unknown_or_unavailable"],
        ["Year", "Securities", "Observations", "Active with data", "Unknown/unavailable"],
        ["r", "r", "r", "r", "r"],
    )

    actions = pd.DataFrame(stats["tables"]["corporate_actions_by_year"])
    _save_csv(actions, "corporate_actions_by_year.csv")
    _latex_table(
        actions,
        GEN_DIR / "table_action_counts.tex",
        ["year", "split_events", "dividend_events"],
        ["Year", "Split events", "Dividend events"],
        ["r", "r", "r"],
    )

    split = pd.DataFrame(stats["tables"]["split_audit"])
    if split.empty:
        split = pd.DataFrame(columns=["ticker", "event_date", "split_factor", "adjusted_return", "raw_diagnostic_return", "implied_factor", "relative_error", "volume_ratio", "status"])
    _save_csv(split, "split_audit.csv")
    if split.empty:
        _write(GEN_DIR / "table_split_audit.tex", r"\multicolumn{7}{c}{No split events in the validated sample.} \\")
    else:
        display = split.copy()
        display["adjusted_return"] = display["adjusted_return"].map(lambda x: _pct(x, 2))
        display["raw_diagnostic_return"] = display["raw_diagnostic_return"].map(lambda x: _pct(x, 2))
        display["volume_ratio"] = display["volume_ratio"].map(lambda x: f"{x:.2f}")
        display["relative_error"] = display["relative_error"].map(lambda x: f"{x:.4f}")
        _latex_table(
            display,
            GEN_DIR / "table_split_audit.tex",
            ["ticker", "event_date", "split_factor", "adjusted_return", "raw_diagnostic_return", "volume_ratio", "status"],
            ["Security", "Event date", "Split", "Adj. return", "Raw return", "Volume ratio", "Status"],
            ["l", "l", "r", "r", "r", "r", "l"],
        )

    dividends = pd.DataFrame(stats["tables"]["dividend_audit"])
    _save_csv(dividends, "dividend_audit.csv")
    if dividends.empty:
        _write(
            GEN_DIR / "table_dividend_audit.tex",
            "\n".join(
                [
                    r"\begin{tabular}{@{}l@{}}",
                    r"\toprule",
                    r"No dividend events in the validated sample. \\",
                    r"\bottomrule",
                    r"\end{tabular}",
                ]
            ),
        )
    else:
        display = dividends.copy()
        display["cash_dividend"] = display["cash_dividend"].map(lambda x: f"{x:.4f}")
        display["adjusted_return"] = display["adjusted_return"].map(
            lambda x: _pct(x, 3)
        )
        display["raw_total_return"] = display["raw_total_return"].map(
            lambda x: _pct(x, 3)
        )
        display["absolute_return_error"] = display["absolute_return_error"].map(
            lambda x: _pct(x, 4)
        )
        _latex_table(
            display,
            GEN_DIR / "table_dividend_audit.tex",
            [
                "ticker",
                "event_date",
                "cash_dividend",
                "adjusted_return",
                "raw_total_return",
                "absolute_return_error",
                "status",
            ],
            [
                "Security",
                "Ex-date",
                "Cash amount",
                "Adj. return",
                "Raw total return",
                "Abs. error",
                "Status",
            ],
            ["l", "l", "r", "r", "r", "r", "l"],
        )

    volume = pd.DataFrame(stats["tables"]["volume_flags"])
    _save_csv(volume, "volume_flags.csv")
    _latex_table(
        volume,
        GEN_DIR / "table_volume_flags.tex",
        ["flag", "observations"],
        ["Flag", "Observations"],
        ["l", "r"],
    )

    missing = pd.DataFrame(
        data["quality_year"][
            ["year", "observations", "missing_adj_close_observations", "missing_volume_observations", "unobserved_sessions_within_security_intervals"]
        ]
    )
    _save_csv(missing, "missingness_by_year.csv")
    _latex_table(
        missing,
        GEN_DIR / "table_missingness_by_year.tex",
        ["year", "missing_adj_close_observations", "missing_volume_observations", "unobserved_sessions_within_security_intervals"],
        ["Year", "Missing adj. close", "Missing volume", "Unobserved sessions"],
        ["r", "r", "r", "r"],
    )

    issues = pd.DataFrame(stats["tables"]["issue_summary"])
    _save_csv(issues, "issue_summary.csv")
    group = issues.groupby("severity", dropna=False)["count"].sum().reset_index().sort_values("count", ascending=False)
    group["scope"] = "universe + validated"
    _latex_table(
        group,
        GEN_DIR / "table_issue_summary.tex",
        ["severity", "count"],
        ["Severity", "Issue count"],
        ["l", "r"],
    )

    identity = (
        issues[issues["scope"].eq("universe")]
        .sort_values("count", ascending=False)
        .head(8)
        .copy()
    )
    identity["issue_type"] = identity["issue_type"].str.replace(
        "_", " ", regex=False
    )
    _latex_table(
        identity,
        GEN_DIR / "table_identity_issues.tex",
        ["severity", "issue_type", "count"],
        ["Severity", "Identity/history issue", "Count"],
        ["l", "l", "r"],
    )

    gap_examples = data["quality_security"]
    gap_examples = gap_examples[gap_examples["max_unobserved_gap_sessions"].fillna(0).gt(0)]
    gap_examples = gap_examples[["security_id", "ticker", "max_unobserved_gap_sessions", "history_status"]].head(10)
    _save_csv(gap_examples, "gap_examples.csv")
    if gap_examples.empty:
        _write(
            GEN_DIR / "table_gap_examples.tex",
            "\n".join(
                [
                    r"\begin{tabular}{@{}l@{}}",
                    r"\toprule",
                    r"No unexpected internal-session gaps were detected in the three-sample audit. \\",
                    r"\bottomrule",
                    r"\end{tabular}",
                ]
            ),
        )
    else:
        _latex_table(
            gap_examples,
            GEN_DIR / "table_gap_examples.tex",
            ["security_id", "ticker", "max_unobserved_gap_sessions", "history_status"],
            ["Security ID", "Ticker", "Maximum gap", "History status"],
        )

    extreme = data["diagnostics"]
    extreme = extreme[extreme["extreme_return_flag"].fillna(False)].copy()
    extreme = extreme.reindex(extreme["diagnostic_pct_return"].abs().sort_values(ascending=False).index).head(12)
    extreme_csv = extreme[["security_id", "ticker", "date", "diagnostic_pct_return", "adj_close"]]
    _save_csv(extreme_csv, "extreme_returns.csv")
    if extreme.empty:
        _write(
            GEN_DIR / "table_extreme_returns.tex",
            "\n".join(
                [
                    r"\begin{tabular}{@{}l@{}}",
                    r"\toprule",
                    r"No sample diagnostic return exceeded the configured 50\% review threshold. \\",
                    r"\bottomrule",
                    r"\end{tabular}",
                ]
            ),
        )
    else:
        display = extreme.copy()
        display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
        display["diagnostic_pct_return"] = display["diagnostic_pct_return"].map(lambda x: _pct(x, 2))
        display["adj_close"] = display["adj_close"].map(lambda x: f"{x:.2f}")
        _latex_table(
            display,
            GEN_DIR / "table_extreme_returns.tex",
            ["ticker", "date", "diagnostic_pct_return", "adj_close"],
            ["Security", "Date", "Diagnostic return", "Adjusted close"],
        )


def _write_figures(stats: dict[str, Any], data: dict[str, Any]) -> None:
    universe_stats = stats["universe"]
    exchange = pd.DataFrame(universe_stats["exchange_counts"].items(), columns=["Exchange", "Securities"])
    exchange = exchange.sort_values("Securities")
    security_type = pd.DataFrame(universe_stats["type_counts"].items(), columns=["Type", "Securities"])
    security_type = security_type.sort_values("Securities")
    security_type["Label"] = security_type["Type"].map(
        {
            "common_stock": "Common stock",
            "common_equity_class": "Common equity class",
            "ordinary_shares": "Ordinary shares",
            "reit_common_stock": "REIT common equity",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for ax, frame, label, color in (
        (axes[0], exchange, "Exchange", COLORS["blue"]),
        (axes[1], security_type, "", COLORS["teal"]),
    ):
        y_values = (
            frame["Label"]
            if label == ""
            else frame.iloc[:, 0]
        )
        counts = frame.iloc[:, 1]
        ax.barh(y_values, counts, color=color)
        ax.set_xlabel("Verified securities")
        if label:
            ax.set_ylabel(label)
        ax.grid(axis="x", alpha=0.8)
        for y, value in enumerate(counts):
            ax.text(
                value + max(counts) * 0.015,
                y,
                f"{int(value):,}",
                va="center",
                fontsize=8,
            )
    axes[0].set_title("Exchange composition")
    axes[1].set_title("Common-equity categories")
    fig.tight_layout()
    _save_figure(fig, "universe_composition.pdf")

    severity = data["universe_issues"]["severity"].value_counts()
    issue_types = data["universe_issues"]["issue_type"].value_counts().head(10).sort_values()
    severity = severity.reindex(
        [level for level in ("high", "medium", "low", "info") if level in severity.index]
    )
    issue_types.index = [value.replace("_", " ") for value in issue_types.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw={"width_ratios": [0.7, 1.4]})
    axes[0].bar(
        severity.index,
        severity.values,
        color=[COLORS["rust"], COLORS["gold"], COLORS["slate"], COLORS["blue"]][
            : len(severity)
        ],
    )
    axes[0].set_title("Universe issue severity")
    axes[0].set_ylabel("Issue groups")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].barh(issue_types.index, issue_types.values, color=COLORS["slate"])
    axes[1].set_title("Most frequent universe issue types")
    axes[1].set_xlabel("Issue groups")
    fig.tight_layout()
    _save_figure(fig, "issue_composition.pdf")

    acq = stats["acquisition"]
    labels = ["Yahoo primary", "EODHD demo sample"]
    success = np.array([acq["yahoo_success"], acq["demo_success"]])
    failed = np.array([acq["yahoo_failed"], acq["demo_failed"]])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.bar(x, success, color=COLORS["teal"], label="success")
    ax.bar(x, failed, bottom=success, color=COLORS["rust"], label="failed")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Securities requested")
    ax.legend()
    ax.set_title("Observed acquisition outcomes")
    for i, total in enumerate(success + failed):
        ax.text(i, total + 0.8, f"{int(total):,}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.8)
    _save_figure(fig, "download_status.pdf")

    year = data["quality_year"].sort_values("year")
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(year["year"], year["observations"], marker="o", color=COLORS["navy"], label="observations")
    ax.set_ylabel("Valid daily observations")
    ax2 = ax.twinx()
    ax2.plot(year["year"], year["securities_with_valid_adj_close"], marker="s", color=COLORS["gold"], label="securities")
    ax2.set_ylabel("Securities (gold)", color=COLORS["gold"])
    ax.tick_params(axis="y", colors=COLORS["navy"])
    ax2.tick_params(axis="y", colors=COLORS["gold"])
    ax.set_title("Validated sample by calendar year")
    ax.grid(axis="y", alpha=0.7)
    fig.tight_layout()
    _save_figure(fig, "observations_by_year.pdf")

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.bar(year["year"].astype(str), year["securities_with_valid_adj_close"], color=COLORS["blue"])
    ax.set_ylim(0, max(5, year["securities_with_valid_adj_close"].max() + 1))
    ax.set_ylabel("Securities")
    ax.set_title("Securities with at least one valid adjusted close")
    _save_figure(fig, "available_securities_by_year.pdf")

    success_quality = data["quality_security"][
        data["quality_security"]["download_status"].eq("success")
    ]
    calendar_rates = pd.to_numeric(
        success_quality["calendar_missing_rate_observed_interval"],
        errors="coerce",
    ).dropna()
    close_rates = pd.to_numeric(
        success_quality["row_missing_adj_close_rate"], errors="coerce"
    ).dropna()
    volume_rates = pd.to_numeric(
        success_quality["row_missing_volume_rate"], errors="coerce"
    ).dropna()
    upper = max(
        0.01,
        float(
            np.nanpercentile(
                pd.concat([calendar_rates, close_rates, volume_rates]),
                99,
            )
        ),
    )
    bins = np.linspace(0, upper, 30)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for values, label, color in (
        (calendar_rates, "unobserved internal sessions", COLORS["blue"]),
        (close_rates, "missing adjusted close", COLORS["gold"]),
        (volume_rates, "missing volume", COLORS["teal"]),
    ):
        ax.hist(
            values.clip(upper=upper),
            bins=bins,
            alpha=0.55,
            label=f"{label} (n={len(values):,})",
            color=color,
        )
    ax.set_xlabel("Missingness rate within observed history")
    ax.set_ylabel("Security count")
    ax.legend(fontsize=7)
    ax.set_title("Missingness distribution across Yahoo histories")
    zero_calendar = int(calendar_rates.eq(0).sum())
    ax.text(
        0.98,
        0.97,
        f"Zero calendar gap: {zero_calendar:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["slate"],
    )
    fig.tight_layout()
    _save_figure(fig, "missingness_distribution.pdf")

    base = data["base"]
    prices = pd.to_numeric(base["adj_close"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    volumes = pd.to_numeric(base["volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.hist(prices, bins=np.logspace(np.log10(max(prices.min(), 0.01)), np.log10(prices.max()), 60), color=COLORS["teal"], alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted close (log scale)")
    ax.set_ylabel("Observation count")
    ax.set_title("Adjusted-close distribution in acquired Yahoo histories")
    _save_figure(fig, "adjusted_close_distribution.pdf")

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.hist(volumes[volumes > 0], bins=np.logspace(np.log10(max(volumes.min(), 1)), np.log10(volumes.max()), 60), color=COLORS["navy"], alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Reported share volume (log scale)")
    ax.set_ylabel("Observation count")
    ax.set_title("Reported share-volume distribution; dollar volume not constructed")
    _save_figure(fig, "volume_distribution_log.pdf")

    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.4), sharex="col")
    tickers = ["AAPL", "MSFT", "TSLA"]
    for col, ticker in enumerate(tickers):
        sample = base[base["ticker"].eq(ticker)].sort_values("date")
        normalized = sample["adj_close"] / sample["adj_close"].iloc[0] * 100
        axes[0, col].plot(sample["date"], normalized, color=COLORS["navy"], linewidth=1.1)
        axes[0, col].set_title(ticker)
        if col == 0:
            axes[0, col].set_ylabel("Index (start=100)")
        axes[0, col].grid(alpha=0.6)
        axes[1, col].plot(sample["date"], sample["volume"], color=COLORS["blue"], linewidth=0.55)
        axes[1, col].set_yscale("log")
        if col == 0:
            axes[1, col].set_ylabel("Share volume")
        axes[1, col].grid(alpha=0.6)
        axes[1, col].xaxis.set_major_locator(mdates.YearLocator(base=2))
        axes[1, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1, col].tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Representative validated histories", y=1.01)
    fig.subplots_adjust(wspace=0.30)
    _save_figure(fig, "representative_histories.pdf")

    actions = pd.DataFrame(stats["tables"]["corporate_actions_by_year"])
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.bar(actions["year"].astype(str), actions["dividend_events"], color=COLORS["gold"], label="dividends")
    ax.bar(actions["year"].astype(str), actions["split_events"], bottom=actions["dividend_events"], color=COLORS["rust"], label="splits")
    ax.set_ylabel("Recorded events")
    ax.set_title("Corporate actions in the validated sample")
    ax.legend()
    _save_figure(fig, "corporate_actions_by_year.pdf")

    split = data["quality_security"]
    success_ids = data["base"]["security_id"].unique()
    raw_events = []
    for security_id in success_ids:
        paths = split.loc[split["security_id"].eq(security_id), "raw_file"]
        if paths.empty:
            continue
        raw = pd.read_parquet(paths.iloc[0]).sort_values("date").reset_index(drop=True)
        raw["ticker"] = raw["ticker_2023"]
        for idx in raw.index[raw["stock_splits"].gt(0)]:
            window = raw.iloc[max(0, idx - 7) : min(len(raw), idx + 8)].copy()
            window["event_index"] = np.arange(-min(7, idx), len(window) - min(7, idx))
            raw_events.append(window)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    if raw_events:
        selected_positions = np.unique(
            np.linspace(0, len(raw_events) - 1, min(12, len(raw_events)), dtype=int)
        )
        for event in [raw_events[position] for position in selected_positions]:
            label = f"{event['ticker'].iloc[0]} ({pd.Timestamp(event.loc[event['event_index'].eq(0), 'date'].iloc[0]).date()})"
            ax.plot(event["event_index"], event["adj_close"] / event.loc[event["event_index"].eq(0), "adj_close"].iloc[0] * 100, marker="o", markersize=2.5, linewidth=1.0, label=label)
    ax.axvline(0, color=COLORS["rust"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Trading sessions relative to split")
    ax.set_ylabel("Adjusted close index (event=100)")
    ax.set_title("Adjusted-price continuity around split events")
    if raw_events:
        ax.legend(fontsize=6, ncol=2)
    _save_figure(fig, "split_adjustment_audit.pdf")

    diag = data["diagnostics"]
    values = pd.to_numeric(diag["diagnostic_pct_return"], errors="coerce").dropna()
    clipped = values.clip(-0.25, 0.25)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.hist(clipped, bins=80, color=COLORS["slate"])
    ax.axvline(-0.50, color=COLORS["rust"], linestyle="--", label="-50% review threshold")
    ax.axvline(0.50, color=COLORS["rust"], linestyle="--", label="+50% review threshold")
    ax.set_xlim(-0.27, 0.27)
    ax.set_xlabel("Diagnostic daily return (display clipped to +/-25%)")
    ax.set_ylabel("Observation count")
    ax.legend()
    ax.set_title("Diagnostic return tails; not a research feature")
    _save_figure(fig, "diagnostic_returns_tail.pdf")


def _macro(name: str, value: Any) -> str:
    text = str(value).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
    return rf"\newcommand{{\{name}}}{{{text}}}"


def _write_generated_text(stats: dict[str, Any]) -> None:
    u = stats["universe"]
    a = stats["acquisition"]
    d = stats["data"]
    audit = stats["auditability"]
    ints = {
        "AuditRows": u["audit_rows"],
        "IncludedSecurities": u["included_securities"],
        "ExcludedAuditRows": u["excluded_audit_rows"],
        "UniverseIssues": u["issue_groups"],
        "CurrentTickerMatches": u["current_ticker_match"],
        "NoCurrentTickerMatches": u["no_current_ticker_match"],
        "TickerChanges": u["recorded_ticker_changes"],
        "ReitCount": u["reit_common_stock"],
        "OrdinarySharesCount": u["ordinary_shares"],
        "CommonClassCount": u["common_equity_class"],
        "AdrExcluded": u["adr_rows"],
        "FundExcluded": u["fund_rows"],
        "PreferredExcluded": u["preferred_rows"],
        "YahooAttempted": a["yahoo_attempted"],
        "YahooSuccess": a["yahoo_success"],
        "YahooFailed": a["yahoo_failed"],
        "DemoSelected": a["demo_selected"],
        "DemoSuccess": a["demo_success"],
        "DemoFailed": a["demo_failed"],
        "ObservedSecurities": d["securities"],
        "UniverseWithoutObservations": d["universe_without_observations"],
        "Observations": d["observations"],
        "MinSessionsPerSecurity": d["valid_sessions_per_security_min"],
        "SessionsPerSecurity": d["valid_sessions_per_security_max"],
        "ExpectedSessions": d["expected_sessions"],
        "MissingAdjClose": d["missing_adj_close"],
        "MissingVolume": d["missing_volume"],
        "NonPositivePrice": d["nonpositive_adj_close"],
        "NegativeVolume": d["negative_volume"],
        "ZeroVolume": d["zero_volume"],
        "UnobservedSessions": d["unobserved_sessions"],
        "SplitEvents": d["split_events"],
        "DividendEvents": d["dividend_events"],
        "SplitFlags": d["split_continuity_flags"],
        "DividendFlags": d["dividend_continuity_flags"],
        "ExtremeReturns": d["extreme_return_rows"],
        "ExtremeVolumeSpikes": d["extreme_volume_spike_rows"],
        "RawAdjustedConflicts": d["raw_adjusted_conflict_rows"],
        "DiagnosticRows": audit["diagnostic_non_null"],
        "DuplicateBaseKeys": audit["duplicate_base_keys"],
        "ValidationIssues": stats["issues"]["data_issue_groups"],
        "CriticalIssues": int(stats["issues"]["severity_counts"].get("critical", 0)),
    }
    decimals = {
        "DiagnosticMin": f"{100 * d['diagnostic_min']:.2f}",
        "DiagnosticMax": f"{100 * d['diagnostic_max']:.2f}",
        "PriceMin": f"{d['price_min']:.2e}",
        "PriceMedian": f"{d['price_median']:.2f}",
        "PriceMax": f"{d['price_max']:.2e}",
        "VolumeMin": _number(d["volume_min"]),
        "VolumeMedian": _number(d["volume_median"]),
        "VolumeMax": _number(d["volume_max"]),
        "PriceStart": stats["scope"]["observation_start"],
        "PriceEnd": stats["scope"]["observation_end"],
    }
    lines = [r"% Generated by src/eda/eda.py; do not edit by hand."]
    lines.extend(_macro(name, _number(value)) for name, value in ints.items())
    lines.extend(_macro(name, value) for name, value in decimals.items())
    _write(GEN_DIR / "macros.tex", "\n".join(lines))

    _write(
        GEN_DIR / "abstract.tex",
        rf"""
This report documents the reconstruction of a historical US common-equity
universe, acquisition of daily Yahoo Finance market data, standardized quality
checks, and exploratory diagnostics for the period 1 January 2016 through
31 December 2025. The verified partial universe contains \IncludedSecurities{{}}
securities derived from \AuditRows{{}} audited SEC cover rows. Yahoo Finance is
the authoritative market-data source for this report: all \YahooAttempted{{}}
verified securities were attempted, \YahooSuccess{{}} produced usable histories,
and \YahooFailed{{}} remained unavailable. The validated Yahoo base contains
\Observations{{}} observations for \ObservedSecurities{{}} securities. The report
treats security identity, historical eligibility, vendor completeness,
missingness, corporate actions, extreme observations, and acquisition
provenance as auditable properties rather than assumptions. It does not
implement a chronological data split, strategy features, model training, or
backtesting.
""",
    )

    _write(
        GEN_DIR / "executive_summary.tex",
        rf"""
The reconstruction identifies \IncludedSecurities{{}} verified common-equity securities from \AuditRows{{}} audited SEC cover rows. The complete exchange-listed denominator for 3 January 2023 remains unavailable, so the reconstruction is explicitly partial and no universe coverage percentage is claimed. Activision Blizzard remains represented through its 2023 symbol, demonstrating that the procedure is not restricted to current survivors.

Yahoo Finance was the primary market-data route. The full verified master was attempted: \YahooSuccess{{}} securities produced usable histories and \YahooFailed{{}} remained unavailable or empty. The validated Yahoo base contains \Observations{{}} rows for \ObservedSecurities{{}} securities over \PriceStart{{}} to \PriceEnd{{}}.

Within observed histories, the raw vendor audit records \MissingAdjClose{{}} missing adjusted-close rows and \MissingVolume{{}} missing volume rows. It also flags \NonPositivePrice{{}} non-positive adjusted closes, \NegativeVolume{{}} negative-volume rows, \ZeroVolume{{}} zero-volume rows, and \ExtremeReturns{{}} diagnostic returns beyond the 50\% review threshold. The vendor reports \SplitEvents{{}} split events and \DividendEvents{{}} dividend events; split continuity checks produce \SplitFlags{{}} flags. These observations are retained or flagged rather than treated as trading information.
""",
    )

    _write(
        GEN_DIR / "universe_method.tex",
        rf"""
The source layer combines SEC Q4 2022 and Q1 2023 company filing indexes with Section 12(b) cover-page inline-XBRL fields. The SEC filing cohort provides evidence that a registrant and covered security existed around the reference date while retaining issuers that later disappeared or changed symbols. This avoids treating a current symbol directory as the historical universe, although the available public filings do not establish an authoritative exchange census.

The build retains \AuditRows{{}} cover-row audit records: \IncludedSecurities{{}} unique verified securities enter the acquisition master and \ExcludedAuditRows{{}} rows remain excluded with a stated reason. Security identity is anchored to SEC CIK where possible, with the 2023 ticker retained separately. Repeated symbols from subsidiary filings are not allowed to create duplicate mappings; evidence-ranked duplicate resolutions and remaining ambiguity are recorded in \UniverseIssues{{}} source issue groups.
""",
    )

    _write(
        GEN_DIR / "universe_results.tex",
        rf"""
The verified master consists of \IncludedSecurities{{}} securities across the three requested exchanges. Of these, \ReitCount{{}} are explicitly identified as REIT common stock, \OrdinarySharesCount{{}} as non-ADR ordinary shares, and \CommonClassCount{{}} as separate common-equity classes. The audit covers exclude \AdrExcluded{{}} ADR, \FundExcluded{{}} fund, and \PreferredExcluded{{}} preferred-security rows based on the observed security title rather than company-name guesses.

The reference-date status is a filing-evidence classification, not a daily exchange-official-list confirmation. Listing and delisting dates remain unknown for most records where the selected sources do not provide reliable interval endpoints. This missing metadata prevents a defensible calculation of full historical-universe coverage.
""",
    )

    _write(
        GEN_DIR / "acquisition_results.tex",
        rf"""
The primary Yahoo implementation attempted all \YahooAttempted{{}} securities in the verified master and obtained \YahooSuccess{{}} successful histories. The remaining \YahooFailed{{}} requests returned no usable observations, generally because the historical symbol was absent or the provider reported a possibly delisted security. Request parameters, timestamps, package versions, and each failure are retained in the acquisition metadata. Empty all-NaN date grids were quarantined rather than recorded as observations.

As a secondary cross-check, \DemoSelected{{}} securities were requested from the EODHD public-demo service; \DemoSuccess{{}} succeeded and \DemoFailed{{}} returned HTTP 403. The main validated results and quality conclusions in this report use Yahoo data, not that restricted secondary sample.
""",
    )

    _write(
        GEN_DIR / "coverage_results.tex",
        rf"""
The verified universe contains \IncludedSecurities{{}} securities. Yahoo returned usable histories for \YahooSuccess{{}} and no acquired observations for \YahooFailed{{}}. The research-facing base contains \Observations{{}} valid rows for \ObservedSecurities{{}} securities over \PriceStart{{}} through \PriceEnd{{}}. Valid-session counts range from \MinSessionsPerSecurity{{}} to \SessionsPerSecurity{{}} per acquired security, reflecting different observation starts and listing histories.

Coverage of the requested historical universe remains unknown as a percentage because no authoritative complete 3 January 2023 exchange census was available. In addition, \UniverseWithoutObservations{{}} master securities have no Yahoo observation. Market-data findings therefore describe the verified acquired subset, not a documented complete population.
""",
    )

    _write(
        GEN_DIR / "missingness_results.tex",
        rf"""
Within observed Yahoo histories, adjusted close is missing \MissingAdjClose{{}} times and volume is missing \MissingVolume{{}} times. Calendar comparisons identify \UnobservedSessions{{}} unobserved internal XNYS sessions across the acquired securities. The research-facing base excludes placeholder and invalid rows, while the raw and standardized audit tables retain them for review.

No missing value was filled or converted to zero. For the \YahooFailed{{}} securities without histories, the cause remains unavailable rather than a trading suspension or delisting determination. Pre-listing and post-delisting boundaries cannot be assigned without verified lifecycle dates.
""",
    )

    _write(
        GEN_DIR / "corporate_action_results.tex",
        rf"""
The acquired Yahoo histories record \SplitEvents{{}} split events and \DividendEvents{{}} dividend events. Raw and adjusted fields are both preserved. Yahoo's nominal Close is itself back-adjusted across splits, so the audit tests continuity between nominal and adjusted closes at event dates while retaining the split multiplier as metadata. \SplitFlags{{}} events remain associated with at least one continuity flag at the three-percent relative-error tolerance. This is a screening result, not proof of a vendor error in every case.

Reported share volume remains in vendor units. The data do not establish that historical volume was normalized for splits, and no split-adjusted volume transformation is performed in this phase.
""",
    )

    _write(
        GEN_DIR / "volume_results.tex",
        rf"""
The acquired histories contain \MissingVolume{{}} missing, \NegativeVolume{{}} negative, and \ZeroVolume{{}} zero-volume rows. At the configured extreme-spike threshold, \ExtremeVolumeSpikes{{}} observations are flagged. The observed positive reported-share-volume range in the research base is \VolumeMin{{}} to \VolumeMax{{}} shares per session, with median \VolumeMedian{{}}.

These are reported share volumes, not dollar trading volumes. No ADV20, relative-volume series, or other strategy-specific feature is constructed. Flags remain review diagnostics and are not automatic exclusion rules.
""",
    )

    _write(
        GEN_DIR / "extreme_results.tex",
        rf"""
The positive adjusted-close sample ranges from \PriceMin{{}} to \PriceMax{{}}, with median \PriceMedian{{}}. The temporary diagnostic return spans \DiagnosticMin{{}}\% to \DiagnosticMax{{}}\% across \DiagnosticRows{{}} non-null observations. \ExtremeReturns{{}} observations cross the absolute 50\% review threshold, and \RawAdjustedConflicts{{}} raw-versus-adjusted continuity conflicts are flagged outside reported corporate-action dates.

The diagnostic figure is descriptive only. The finalized \texttt{{pct\_return}} column is null for every base row, so no diagnostic return enters a research feature, label, split, model, or backtest.
""",
    )

    _write(
        GEN_DIR / "identity_bias_results.tex",
        rf"""
\CurrentTickerMatches{{}} included securities match a current SEC ticker, while \NoCurrentTickerMatches{{}} do not. A missing current match can reflect a merger, delisting, ticker change, or SEC reference limitation; it is not by itself a delisting determination. \TickerChanges{{}} included rows record an apparent 2023-to-current symbol difference.

Subsidiary and partnership cover duplicates are resolved before creating the acquisition master, while unresolved historical collisions remain visible in the universe issue register. The retained Activision Blizzard example demonstrates delisted-name representation. Nevertheless, SEC-filing coverage is non-random and cannot be assumed to reproduce the complete historical list \citep{{brown1992survivorship,shumway1997delisting}}.
""",
    )

    _write(
        GEN_DIR / "cleaning_decisions.tex",
        rf"""
Validation removed \DuplicateBaseKeys{{}} duplicate security-date keys from the research-facing base after deterministic ordering; no duplicate key remains. Raw observations and separate standardized audit data remain available. Missing values were not zero-filled, and no price or volume was winsorized, clipped, or deleted merely because it was extreme.

The split-continuity test was made vendor-aware: Yahoo's nominal Close is already split-continuous and is checked against a unit implied factor, while an unadjusted close series is checked against the event multiplier. The final base retains a null \texttt{{pct\_return}} column as required. The validation issue register contains \ValidationIssues{{}} grouped findings and \CriticalIssues{{}} critical groups; it separates acquisition failures from observation-level quality checks.
""",
    )

    _write(
        GEN_DIR / "limitations.tex",
        r"""
Yahoo histories were obtained for \YahooSuccess{{}} of \YahooAttempted{{}} verified securities, leaving \YahooFailed{{}} without observations. This acquisition gap need not be random: removed symbols, reused tickers, and rename history can be systematically absent from the current Yahoo namespace. Historical-universe reconstruction is also incomplete because SEC filing evidence is not an official listing census and some filings lack extractable cover fields.

The available metadata cannot resolve every listing and delisting interval, ticker history, or security classification. The acquired panel still contains gaps, combined corporate-action review flags, raw-versus-adjusted conflicts, extreme observations, and zero-volume sessions. None of these findings justifies zero-filling or deleting observations merely because they are unusual.
""",
    )

    _write(
        GEN_DIR / "conclusions.tex",
        rf"""
The project delivers a reproducible partial historical universe of \IncludedSecurities{{}} unique common-equity securities, a completed Yahoo attempt over all of them, \YahooSuccess{{}} acquired histories, a validated \Observations{{}}-row research-facing base, structured identity and quality issue registers, and a compiled quality-assessment report. The missing \YahooFailed{{}} histories and unavailable complete-listing denominator prevent a claim of full historical coverage.

The correct interpretation is a substantial verified Yahoo dataset with explicit coverage uncertainty, not a complete 2016--2025 US equity census. Later research must account for the SEC-derived master's non-census nature, unavailable delisted histories, unresolved identifiers, missing lifecycle dates, and the possibility that observed and unobserved firms differ materially.
""",
    )

    _write(
        GEN_DIR / "configuration.tex",
        rf"""
The reference date is fixed at 3 January 2023 and the requested observation interval is 1 January 2016 through 31 December 2025. SEC access uses the \texttt{{SEC\_USER\_AGENT}} environment variable without embedded personal credentials. Yahoo requests use \texttt{{auto\_adjust=False}}, corporate actions enabled, bounded retries, and per-security checkpoints.

The optional public-demo cross-check remains configured separately as \texttt{{secondary\_sample\_only}} with \texttt{{full\_universe\_replacement=false}}. The primary validated evidence in this report is produced from the Yahoo raw directory and its manifests.
""",
    )

    _write(
        GEN_DIR / "reproduction_commands.tex",
        r"""
\begin{verbatim}
conda env create -f environment.yml
conda activate dsa5205_proj2
export SEC_USER_AGENT='Academic project name and contact email'
python scripts/build_universe.py
python scripts/download_data.py \
  --universe data/universe/security_master.parquet \
  --batch-size 50
python scripts/validate_data.py \
  --config configs/acquisition.yaml \
  --raw-dir data/raw/yahoo \
  --master data/universe/security_master.parquet \
  --output-dir data/validated \
  --metadata-dir data/metadata \
  --manifest data/metadata/download_manifest.csv \
  --failures data/metadata/download_failures.csv
python scripts/run_eda.py
cd reports/latex && latexmk -pdf -interaction=nonstopmode main.tex
python -m pytest -q
\end{verbatim}
""",
    )


def run(config_path: str | Path) -> dict[str, Any]:
    configure_logging(PROJECT_ROOT / "logs" / "eda.log")
    _figure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_inputs()
    stats = _compute_statistics(data)
    _write_tables(stats, data)
    _write_figures(stats, data)
    atomic_write_json(stats, STATS_PATH)
    _write_generated_text(stats)
    return {
        "statistics": str(STATS_PATH),
        "figures": sorted(path.name for path in FIG_DIR.glob("*.pdf")),
        "tables": sorted(path.name for path in TABLE_DIR.glob("*.csv")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acquisition.yaml")
    args = parser.parse_args(argv)
    config_path = project_path(args.config)
    result = run(config_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
