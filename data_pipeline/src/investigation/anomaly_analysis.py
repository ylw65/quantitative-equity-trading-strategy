"""Investigate extreme prices, returns, and corporate-action continuity.

The investigation is read-only with respect to vendor checkpoints. It creates
separate evidence and quarantine-candidate tables so later analytical code can
exclude uncertain windows without losing the raw observations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, atomic_write_json, configure_logging, load_config


OUTPUT_DIR = PROJECT_ROOT / "data" / "investigation" / "anomalies"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "investigation"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "investigation"

COLORS = {
    "navy": "#19324D",
    "blue": "#3F6F8F",
    "teal": "#4E8C86",
    "gold": "#B28A48",
    "rust": "#A45D4F",
    "slate": "#65727A",
}


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=False)
    frame.to_csv(TABLE_DIR / name, index=False)
    return OUTPUT_DIR / name


def _event_window(
    frame: pd.DataFrame,
    index: int,
    before: int = 5,
    after: int = 5,
) -> pd.DataFrame:
    start = max(0, index - before)
    end = min(len(frame), index + after + 1)
    window = frame.iloc[start:end].copy()
    window["relative_session"] = np.arange(start - index, end - index)
    return window


def classify_extreme_return(
    previous_adj_close: float,
    diagnostic_return: float,
    *,
    same_day_split: bool,
    same_day_dividend: bool,
    split_within_five_sessions: bool,
    vendor_repaired: bool,
) -> str:
    """Classify an extreme return using reproducible evidence precedence."""
    if not math.isfinite(float(previous_adj_close)):
        return "missing_previous_price"
    if float(previous_adj_close) <= 0.01:
        return "near_zero_denominator"
    if same_day_split and same_day_dividend:
        return "simultaneous_split_and_distribution"
    if same_day_split:
        return "same_day_split_adjustment"
    if same_day_dividend:
        return "same_day_dividend_adjustment"
    if split_within_five_sessions:
        return "split_adjustment_near_event"
    if vendor_repaired and abs(float(diagnostic_return)) >= 5:
        return "vendor_repaired_price_discontinuity"
    if abs(float(diagnostic_return)) >= 10:
        return "unresolved_extreme_discontinuity"
    return "retained_extreme_market_move"


def classify_price_level(
    maximum_adjusted_close: float,
    minimum_adjusted_close: float,
    median_adjusted_close: float,
    reverse_split_count: int,
) -> str:
    """Classify abnormal adjusted-price levels without treating them as invalid."""
    if maximum_adjusted_close > 1_000_000:
        if reverse_split_count > 0:
            return "consistent_with_reverse_split_adjustment_history"
        return "unresolved_extreme_adjusted_price_level"
    if minimum_adjusted_close < 0.0001:
        return "sub_cent_adjusted_price_observation"
    if (
        median_adjusted_close > 0
        and maximum_adjusted_close / median_adjusted_close > 1_000_000
    ):
        return "extreme_adjustment_range"
    return "not_classified_extreme"


def classify_corporate_action(
    *,
    event_kind: str,
    dividend_fraction: float,
    split_same_day: bool,
) -> tuple[str, str, str]:
    """Return classification, resolution status, and confidence."""
    if split_same_day and dividend_fraction > 0:
        return (
            "simultaneous_split_and_distribution",
            "resolved_methodology",
            "high",
        )
    if event_kind == "split":
        return (
            "split_adjustment_discontinuity",
            "unresolved",
            "medium",
        )
    if dividend_fraction >= 0.20:
        return (
            "probable_special_distribution_or_spinoff",
            "partially_resolved",
            "medium",
        )
    if dividend_fraction >= 0.05:
        return (
            "large_distribution_or_adjustment_artifact",
            "partially_resolved",
            "low",
        )
    return (
        "ordinary_dividend_adjustment_mismatch",
        "unresolved",
        "medium",
    )


def _load_quality() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "validated" / "data_quality_by_security.csv"
    return pd.read_csv(path, low_memory=False)


def _raw_path(row: pd.Series) -> Path:
    path = Path(str(row["raw_file"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_raw(row: pd.Series) -> pd.DataFrame:
    raw = pd.read_parquet(_raw_path(row)).sort_values("date").reset_index(drop=True)
    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw checkpoint missing fields {missing}: {row['raw_file']}")
    return raw


def reconcile_corporate_actions(
    quality: pd.DataFrame,
    split_tolerance: float,
    dividend_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute and classify flagged split and dividend events."""
    split_rows: list[dict[str, Any]] = []
    dividend_rows: list[dict[str, Any]] = []
    windows: list[pd.DataFrame] = []

    # Recompute the small native-continuity discrepancy set even when the
    # current validator has already moved combined split/distribution events
    # out of its simple-split issue category.
    split_securities = quality[
        quality["split_events"].fillna(0).gt(0)
    ]
    for _, metadata in split_securities.iterrows():
        raw = _read_raw(metadata)
        previous = raw.shift(1)
        raw_ratio = previous["close"] / raw["close"]
        adjusted_gross = raw["adj_close"] / previous["adj_close"]
        implied = raw_ratio * adjusted_gross
        native_continuity = implied.sub(1).abs()
        mask = raw["stock_splits"].gt(0) & native_continuity.gt(split_tolerance)
        for index in raw.index[mask]:
            row = raw.loc[index]
            old = previous.loc[index]
            dividend_fraction = (
                float(row["dividends"]) / float(old["close"])
                if pd.notna(old["close"]) and float(old["close"]) > 0
                else 0.0
            )
            classification, status, confidence = classify_corporate_action(
                event_kind="split",
                dividend_fraction=dividend_fraction,
                split_same_day=float(row["stock_splits"]) > 0,
            )
            split_rows.append(
                {
                    "security_id": metadata["security_id"],
                    "ticker": metadata["ticker"],
                    "event_date": pd.Timestamp(row["date"]).date().isoformat(),
                    "previous_date": pd.Timestamp(old["date"]).date().isoformat(),
                    "previous_close": float(old["close"]),
                    "close": float(row["close"]),
                    "previous_adj_close": float(old["adj_close"]),
                    "adj_close": float(row["adj_close"]),
                    "stock_split": float(row["stock_splits"]),
                    "dividend": float(row["dividends"]),
                    "vendor_repaired": _bool(row.get("repaired", False)),
                    "raw_gross_return": float(row["close"] / old["close"] - 1),
                    "adjusted_gross_return": float(adjusted_gross.loc[index] - 1),
                    "implied_yahoo_continuity_factor": float(implied.loc[index]),
                    "relative_error_from_unit": float(native_continuity.loc[index]),
                    "original_flag": "split_adjustment_continuity",
                    "final_classification": classification,
                    "resolution_status": status,
                    "confidence": confidence,
                    "evidence": (
                        "Recomputed from raw Close and Adj Close. Same-day split and "
                        f"dividend metadata: split={float(row['stock_splits'])}, "
                        f"dividend={float(row['dividends'])}."
                    ),
                    "uncertainty": (
                        "Yahoo does not provide a structured distribution type; "
                        "special distribution versus spin-off remains unresolved."
                    ),
                    "recommended_action": (
                        "Exclude combined-action dates from the simple split "
                        "continuity test and retain the event for audit."
                    ),
                }
            )
            window = _event_window(raw, int(index))
            window["ticker"] = metadata["ticker"]
            window["security_id"] = metadata["security_id"]
            window["event_kind"] = "split"
            window["event_window_date"] = pd.Timestamp(row["date"]).date().isoformat()
            windows.append(window)

    dividend_securities = quality[
        quality["dividend_continuity_flags"].fillna(0).gt(0)
    ]
    for _, metadata in dividend_securities.iterrows():
        raw = _read_raw(metadata)
        previous = raw.shift(1)
        adjusted_gross = raw["adj_close"] / previous["adj_close"]
        raw_total_return = (
            raw["close"] + raw["dividends"]
        ).div(previous["close"]).sub(1)
        error = adjusted_gross.sub(1).sub(raw_total_return).abs()
        mask = raw["dividends"].gt(0) & error.gt(dividend_tolerance)
        for index in raw.index[mask]:
            row = raw.loc[index]
            old = previous.loc[index]
            dividend_fraction = (
                float(row["dividends"]) / float(old["close"])
                if pd.notna(old["close"]) and float(old["close"]) > 0
                else 0.0
            )
            split_same_day = float(row["stock_splits"]) > 0
            classification, status, confidence = classify_corporate_action(
                event_kind="dividend",
                dividend_fraction=dividend_fraction,
                split_same_day=split_same_day,
            )
            dividend_rows.append(
                {
                    "security_id": metadata["security_id"],
                    "ticker": metadata["ticker"],
                    "event_date": pd.Timestamp(row["date"]).date().isoformat(),
                    "previous_date": pd.Timestamp(old["date"]).date().isoformat(),
                    "previous_close": float(old["close"]),
                    "close": float(row["close"]),
                    "previous_adj_close": float(old["adj_close"]),
                    "adj_close": float(row["adj_close"]),
                    "dividend": float(row["dividends"]),
                    "dividend_fraction_of_previous_close": dividend_fraction,
                    "stock_split": float(row["stock_splits"]),
                    "vendor_repaired": _bool(row.get("repaired", False)),
                    "raw_total_return": float(raw_total_return.loc[index]),
                    "adjusted_return": float(adjusted_gross.loc[index] - 1),
                    "absolute_return_error": float(error.loc[index]),
                    "original_flag": "dividend_adjustment_continuity",
                    "final_classification": classification,
                    "resolution_status": status,
                    "confidence": confidence,
                    "evidence": (
                        "Recomputed from raw Close, Adj Close, and reported cash "
                        "distribution; event magnitude relative to prior close is "
                        f"{dividend_fraction:.4f}."
                    ),
                    "uncertainty": (
                        "Yahoo labels spin-offs, special distributions, and cash "
                        "dividends in one field without an event-type flag."
                    ),
                    "recommended_action": (
                        "Retain event; use a combined-action adjustment only after "
                        "independent event-type confirmation."
                    ),
                }
            )
            window = _event_window(raw, int(index))
            window["ticker"] = metadata["ticker"]
            window["security_id"] = metadata["security_id"]
            window["event_kind"] = "dividend"
            window["event_window_date"] = pd.Timestamp(row["date"]).date().isoformat()
            windows.append(window)

    split_frame = pd.DataFrame(split_rows).sort_values(
        ["event_date", "ticker"], kind="stable"
    )
    dividend_frame = pd.DataFrame(dividend_rows).sort_values(
        ["event_date", "ticker"], kind="stable"
    )
    evidence_path = OUTPUT_DIR / "external_evidence.csv"
    if evidence_path.exists():
        evidence = pd.read_csv(evidence_path, dtype=str)
        for frame in (split_frame, dividend_frame):
            if frame.empty:
                continue
            enriched = frame.merge(
                evidence[
                    [
                        "ticker",
                        "event_date",
                        "source_organization",
                        "filing_form",
                        "filing_date",
                        "accession_number",
                        "url",
                        "verified_fact",
                        "confidence",
                    ]
                ].rename(
                    columns={
                        "confidence": "external_confidence",
                        "url": "external_evidence_url",
                        "verified_fact": "external_verified_fact",
                    }
                ),
                on=["ticker", "event_date"],
                how="left",
            )
            frame.drop(frame.index, inplace=True)
            for column in enriched.columns:
                frame[column] = enriched[column].values
            verified = frame["external_verified_fact"].fillna("").ne("")
            frame.loc[verified, "confidence"] = frame.loc[
                verified, "external_confidence"
            ]
            frame.loc[verified, "resolution_status"] = "resolved_event_type"
            frame.loc[verified, "evidence"] = (
                frame.loc[verified, "evidence"]
                + " SEC filing evidence: "
                + frame.loc[verified, "external_verified_fact"]
            )
    window_frame = (
        pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    )
    return split_frame, dividend_frame, window_frame


def audit_extreme_returns(
    quality: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Recalculate extreme returns and attach event/evidence context."""
    successes = quality[quality["download_status"].eq("success")]
    rows: list[dict[str, Any]] = []
    for _, metadata in successes.iterrows():
        raw = _read_raw(metadata)
        previous = raw.shift(1)
        diagnostic_return = raw["adj_close"] / previous["adj_close"] - 1
        diagnostic_return = diagnostic_return.where(
            previous["adj_close"].gt(0) & raw["adj_close"].gt(0)
        )
        split_indexes = np.flatnonzero(raw["stock_splits"].gt(0).to_numpy())
        mask = diagnostic_return.abs().gt(threshold).fillna(False)
        if not mask.any():
            continue
        for index in raw.index[mask]:
            row = raw.loc[index]
            old = previous.loc[index]
            split_distance = (
                min(abs(int(index) - int(split_index)) for split_index in split_indexes)
                if len(split_indexes)
                else np.nan
            )
            ret = float(diagnostic_return.loc[index])
            same_split = float(row["stock_splits"]) > 0
            same_dividend = float(row["dividends"]) > 0
            repaired = _bool(row.get("repaired", False)) or _bool(
                old.get("repaired", False)
            )
            category = classify_extreme_return(
                float(old["adj_close"]),
                ret,
                same_day_split=same_split,
                same_day_dividend=same_dividend,
                split_within_five_sessions=(
                    pd.notna(split_distance) and split_distance <= 5
                ),
                vendor_repaired=repaired,
            )
            quarantine = (
                category in {
                    "near_zero_denominator",
                    "unresolved_extreme_discontinuity",
                    "vendor_repaired_price_discontinuity",
                }
                or abs(ret) >= 10
            )
            rows.append(
                {
                    "security_id": metadata["security_id"],
                    "ticker": metadata["ticker"],
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "previous_date": pd.Timestamp(old["date"]).date().isoformat(),
                    "previous_adj_close": float(old["adj_close"]),
                    "adj_close": float(row["adj_close"]),
                    "diagnostic_return": ret,
                    "previous_close": float(old["close"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]) if pd.notna(row["volume"]) else np.nan,
                    "dividend": float(row["dividends"]),
                    "stock_split": float(row["stock_splits"]),
                    "nearby_split_distance_sessions": split_distance,
                    "vendor_repaired": repaired,
                    "classification": category,
                    "recommended_for_return_analysis": not quarantine,
                    "recommended_quarantine_for_return_diagnostics": quarantine,
                    "evidence": (
                        f"Previous Adj Close={float(old['adj_close']):.10g}; "
                        f"current Adj Close={float(row['adj_close']):.10g}; "
                        f"return={ret:.6g}."
                    ),
                    "uncertainty": (
                        "Vendor data cannot always distinguish a genuine microcap "
                        "move from a bad quote or adjustment discontinuity."
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("diagnostic_return", key=lambda s: s.abs(), ascending=False)


def audit_extreme_prices(quality: pd.DataFrame) -> pd.DataFrame:
    """Summarize abnormal positive adjusted-price levels by security."""
    successes = quality[quality["download_status"].eq("success")]
    rows: list[dict[str, Any]] = []
    for _, metadata in successes.iterrows():
        raw = _read_raw(metadata)
        prices = raw["adj_close"].where(raw["adj_close"].gt(0)).dropna()
        if prices.empty:
            continue
        maximum = float(prices.max())
        minimum = float(prices.min())
        median = float(prices.median())
        if maximum <= 1_000_000 and minimum >= 0.0001:
            continue
        max_index = int(prices.idxmax())
        min_index = int(prices.idxmin())
        split_factors = raw.loc[raw["stock_splits"].gt(0), "stock_splits"]
        reverse_count = int(split_factors.lt(1).sum())
        cumulative_split = float(np.prod(split_factors)) if len(split_factors) else 1.0
        classification = classify_price_level(
            maximum,
            minimum,
            median,
            reverse_count,
        )
        rows.append(
            {
                "security_id": metadata["security_id"],
                "ticker": metadata["ticker"],
                "first_date": pd.Timestamp(raw["date"].min()).date().isoformat(),
                "last_date": pd.Timestamp(raw["date"].max()).date().isoformat(),
                "valid_price_rows": int(len(prices)),
                "min_adjusted_close": minimum,
                "min_date": pd.Timestamp(raw.loc[min_index, "date"]).date().isoformat(),
                "median_adjusted_close": median,
                "max_adjusted_close": maximum,
                "max_date": pd.Timestamp(raw.loc[max_index, "date"]).date().isoformat(),
                "max_to_median_ratio": (
                    maximum / median if median > 0 else np.nan
                ),
                "split_events": int(len(split_factors)),
                "reverse_split_events": reverse_count,
                "cumulative_split_product": cumulative_split,
                "zero_volume_share": float(raw["volume"].eq(0).mean()),
                "classification": classification,
                "confidence": (
                    "medium"
                    if classification
                    == "consistent_with_reverse_split_adjustment_history"
                    else "low"
                ),
                "evidence": (
                    "Highest and lowest positive adjusted closes with the vendor "
                    "split history and zero-volume share."
                ),
                "uncertainty": (
                    "The adjusted level can be arithmetically consistent with "
                    "reverse splits while still being unsuitable for level-based "
                    "feature transformations."
                ),
                "recommended_action": (
                    "Retain raw observations; quarantine the security from "
                    "unlogged level-based transforms pending review."
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("max_adjusted_close", ascending=False)


def build_quarantine_register(
    extreme_returns: pd.DataFrame,
    extreme_prices: pd.DataFrame,
    split_events: pd.DataFrame,
    dividend_events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not extreme_returns.empty:
        selected = extreme_returns[
            extreme_returns[
                "recommended_quarantine_for_return_diagnostics"
            ].astype(bool)
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "security_id": row["security_id"],
                    "ticker": row["ticker"],
                    "start_date": row["previous_date"],
                    "end_date": row["date"],
                    "scope": "return_diagnostics",
                    "reason": row["classification"],
                    "evidence": row["evidence"],
                    "raw_preserved": True,
                    "base_rows_modified": False,
                }
            )
    if not extreme_prices.empty:
        selected_prices = extreme_prices[
            extreme_prices["classification"].isin(
                {
                    "consistent_with_reverse_split_adjustment_history",
                    "unresolved_extreme_adjusted_price_level",
                    "sub_cent_adjusted_price_observation",
                    "extreme_adjustment_range",
                }
            )
        ]
        for _, row in selected_prices.iterrows():
            rows.append(
                {
                    "security_id": row["security_id"],
                    "ticker": row["ticker"],
                    "start_date": row["first_date"],
                    "end_date": row["last_date"],
                    "scope": "price_level_transformations",
                    "reason": row["classification"],
                    "evidence": row["evidence"],
                    "raw_preserved": True,
                    "base_rows_modified": False,
                }
            )
    for frame, scope in (
        (split_events, "split_continuity"),
        (dividend_events, "dividend_continuity"),
    ):
        if frame.empty:
            continue
        unresolved = frame[
            ~frame["resolution_status"].isin(
                {"resolved_methodology", "resolved_event_type"}
            )
        ]
        for _, row in unresolved.iterrows():
            rows.append(
                {
                    "security_id": row["security_id"],
                    "ticker": row["ticker"],
                    "start_date": row["previous_date"],
                    "end_date": row["event_date"],
                    "scope": scope,
                    "reason": row["final_classification"],
                    "evidence": row["evidence"],
                    "raw_preserved": True,
                    "base_rows_modified": False,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.drop_duplicates(
        ["security_id", "start_date", "end_date", "scope", "reason"]
    ).sort_values(["scope", "ticker", "start_date"])


def summarize_anomaly_issues(
    extreme_returns: pd.DataFrame,
    extreme_prices: pd.DataFrame,
    split_events: pd.DataFrame,
    dividend_events: pd.DataFrame,
) -> pd.DataFrame:
    """Create security-level review issues for downstream integration."""
    rows: list[dict[str, Any]] = []
    if not extreme_returns.empty:
        for (security_id, ticker, classification), group in extreme_returns.groupby(
            ["security_id", "ticker", "classification"], dropna=False
        ):
            severity = (
                "high"
                if classification
                in {
                    "near_zero_denominator",
                    "vendor_repaired_price_discontinuity",
                    "unresolved_extreme_discontinuity",
                }
                else "medium"
            )
            resolved = classification in {
                "simultaneous_split_and_distribution",
                "same_day_split_adjustment",
                "same_day_dividend_adjustment",
                "split_adjustment_near_event",
            }
            rows.append(
                {
                    "issue_id": "",
                    "severity": severity,
                    "issue_type": "extreme_return_evidence",
                    "scope": "P1_2_price_return",
                    "security_id": security_id,
                    "ticker": ticker,
                    "affected_rows": len(group),
                    "first_date": group["date"].min(),
                    "last_date": group["date"].max(),
                    "status": "resolved" if resolved else "partially_resolved",
                    "resolution": classification,
                    "evidence": (
                        "Chronological raw/Adj Close audit; see extreme_return_cases.csv."
                    ),
                    "uncertainty": (
                        "Genuine microcap moves and vendor discontinuities can share "
                        "the same magnitude."
                    ),
                }
            )
    if not extreme_prices.empty:
        for (security_id, ticker, classification), group in extreme_prices.groupby(
            ["security_id", "ticker", "classification"], dropna=False
        ):
            rows.append(
                {
                    "issue_id": "",
                    "severity": "medium",
                    "issue_type": "extreme_adjusted_price_level",
                    "scope": "P1_2_price_return",
                    "security_id": security_id,
                    "ticker": ticker,
                    "affected_rows": int(group["valid_price_rows"].sum()),
                    "first_date": group["min_date"].min(),
                    "last_date": group["max_date"].max(),
                    "status": "partially_resolved",
                    "resolution": classification,
                    "evidence": group["evidence"].iloc[0],
                    "uncertainty": group["uncertainty"].iloc[0],
                }
            )
    for frame, issue_type in (
        (split_events, "split_continuity_discrepancy"),
        (dividend_events, "dividend_continuity_discrepancy"),
    ):
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            rows.append(
                {
                    "issue_id": "",
                    "severity": "medium",
                    "issue_type": issue_type,
                    "scope": "P1_4_corporate_actions",
                    "security_id": row["security_id"],
                    "ticker": row["ticker"],
                    "affected_rows": 1,
                    "first_date": row["event_date"],
                    "last_date": row["event_date"],
                    "status": row["resolution_status"],
                    "resolution": row["final_classification"],
                    "evidence": row["evidence"],
                    "uncertainty": row["uncertainty"],
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["scope", "severity", "ticker", "first_date"], kind="stable"
    ).reset_index(drop=True)
    result["issue_id"] = [
        f"ANOM-{number:05d}" for number in range(1, len(result) + 1)
    ]
    return result


def _save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(
    extreme_returns: pd.DataFrame,
    extreme_prices: pd.DataFrame,
    split_events: pd.DataFrame,
    dividend_events: pd.DataFrame,
    event_windows: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.edgecolor": "#68777E",
            "pdf.fonttype": 42,
            "figure.dpi": 120,
        }
    )

    categories = extreme_returns["classification"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.barh(categories.index.str.replace("_", " "), categories.values, color=COLORS["rust"])
    ax.set_xlabel("Extreme diagnostic-return rows")
    ax.set_title("Extreme-return evidence categories")
    ax.grid(axis="x", alpha=0.3)
    _save_figure(fig, "anomaly_extreme_return_categories.pdf")

    action_counts = pd.DataFrame(
        {
            "category": (
                list(split_events.get("final_classification", []))
                + list(dividend_events.get("final_classification", []))
            )
        }
    )
    if not action_counts.empty:
        counts = (
            action_counts["category"]
            .value_counts()
            .rename_axis("category")
            .reset_index(name="events")
            .sort_values("events")
        )
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.barh(
            counts["category"].str.replace("_", " "),
            counts["events"],
            color=COLORS["gold"],
        )
        ax.set_xlabel("Flagged events")
        ax.set_title("Corporate-action flag reconciliation")
        ax.grid(axis="x", alpha=0.3)
        _save_figure(fig, "anomaly_corporate_action_reconciliation.pdf")

    if not extreme_prices.empty:
        plot = extreme_prices.head(20).copy()
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        sizes = 20 + plot["zero_volume_share"].fillna(0) * 100
        ax.scatter(
            plot["reverse_split_events"],
            plot["max_adjusted_close"],
            s=sizes,
            c=plot["zero_volume_share"],
            cmap="viridis",
            alpha=0.8,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Reverse-split events in checkpoint")
        ax.set_ylabel("Maximum adjusted close (log scale)")
        ax.set_title("Extreme adjusted-price levels and reverse-split history")
        for _, row in plot.iterrows():
            ax.annotate(
                row["ticker"],
                (row["reverse_split_events"], row["max_adjusted_close"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )
        _save_figure(fig, "anomaly_extreme_price_levels.pdf")

    selected: list[tuple[str, str]] = []
    candidates = [
        ("VAXX", "2025-03-26"),
        ("WHLR", "2017-02-27"),
        ("DHR", "2016-07-05"),
        ("KDP", "2018-07-10"),
    ]
    available_tickers = set(quality.loc[quality["download_status"].eq("success"), "ticker"])
    for ticker, event_date in candidates:
        if ticker in available_tickers:
            selected.append((ticker, event_date))
    if selected:
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2))
        for ax, (ticker, event_date) in zip(axes.flat, selected):
            window = event_windows[
                event_windows["ticker"].eq(ticker)
                & event_windows["event_window_date"].eq(event_date)
            ]
            if window.empty:
                metadata = quality[quality["ticker"].eq(ticker)].iloc[0]
                raw = _read_raw(metadata)
                matches = raw.index[
                    pd.to_datetime(raw["date"]).eq(pd.Timestamp(event_date))
                ]
                if len(matches):
                    window = _event_window(raw, int(matches[0]))
            ax.plot(
                window["relative_session"],
                window["close"],
                marker="o",
                ms=4,
                label="Close",
                color=COLORS["slate"],
            )
            ax.plot(
                window["relative_session"],
                window["adj_close"],
                marker="o",
                ms=4,
                label="Adj Close",
                color=COLORS["blue"],
            )
            if window["close"].min() > 0 and (
                window["close"].max() / window["close"].min() > 100
            ):
                ax.set_yscale("log")
            ax.axvline(0, color=COLORS["rust"], linestyle="--", linewidth=0.8)
            ax.set_title(f"{ticker}: {event_date}")
            ax.set_xlabel("Sessions relative to event")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.tight_layout()
        _save_figure(fig, "anomaly_representative_windows.pdf")


def run(config_path: str | Path) -> dict[str, Any]:
    configure_logging(PROJECT_ROOT / "logs" / "anomaly_investigation.log")
    config = load_config(config_path)
    quality = _load_quality()
    split_tolerance = float(config["validation"]["split_relative_tolerance"])
    dividend_tolerance = float(config["validation"]["dividend_adjustment_tolerance"])
    return_threshold = float(config["validation"]["extreme_return_abs_threshold"])

    split_events, dividend_events, windows = reconcile_corporate_actions(
        quality,
        split_tolerance,
        dividend_tolerance,
    )
    extreme_returns = audit_extreme_returns(quality, return_threshold)
    extreme_prices = audit_extreme_prices(quality)
    quarantine = build_quarantine_register(
        extreme_returns,
        extreme_prices,
        split_events,
        dividend_events,
    )
    issues = summarize_anomaly_issues(
        extreme_returns,
        extreme_prices,
        split_events,
        dividend_events,
    )

    if not windows.empty:
        windows = windows.sort_values(
            ["security_id", "event_kind", "event_window_date", "date"],
            kind="stable",
        )
        windows["date"] = pd.to_datetime(windows["date"]).dt.strftime("%Y-%m-%d")

    _write_csv(
        split_events,
        "corporate_action_split_reconciliation.csv",
    )
    _write_csv(
        dividend_events,
        "corporate_action_dividend_reconciliation.csv",
    )
    _write_csv(extreme_returns, "extreme_return_cases.csv")
    _write_csv(extreme_prices, "extreme_price_cases.csv")
    _write_csv(quarantine, "analysis_quarantine_candidates.csv")
    _write_csv(issues, "anomaly_issues.csv")
    _write_csv(windows, "corporate_action_event_windows.csv")

    top_cases = extreme_returns.head(25).copy()
    _write_csv(top_cases, "extreme_return_representative_cases.csv")
    make_figures(
        extreme_returns,
        extreme_prices,
        split_events,
        dividend_events,
        windows,
        quality,
    )

    summary = {
        "source": "Yahoo raw checkpoints; no raw or base rows modified",
        "split_flags_original": int(
            quality["split_continuity_flags"].fillna(0).sum()
        ),
        "split_flags_reconciled": int(len(split_events)),
        "split_classifications": (
            split_events["final_classification"].value_counts().to_dict()
            if not split_events.empty
            else {}
        ),
        "dividend_flags_original": int(
            quality["dividend_continuity_flags"].fillna(0).sum()
        ),
        "dividend_flags_reconciled": int(len(dividend_events)),
        "dividend_classifications": (
            dividend_events["final_classification"].value_counts().to_dict()
            if not dividend_events.empty
            else {}
        ),
        "extreme_return_rows": int(len(extreme_returns)),
        "extreme_return_classifications": (
            extreme_returns["classification"].value_counts().to_dict()
            if not extreme_returns.empty
            else {}
        ),
        "extreme_return_quarantine_candidates": int(
            extreme_returns[
                "recommended_quarantine_for_return_diagnostics"
            ].fillna(False).sum()
            if not extreme_returns.empty
            else 0
        ),
        "extreme_price_securities": int(len(extreme_prices)),
        "extreme_price_classifications": (
            extreme_prices["classification"].value_counts().to_dict()
            if not extreme_prices.empty
            else {}
        ),
        "quarantine_rows": int(len(quarantine)),
        "quarantine_securities": int(quarantine["security_id"].nunique())
        if not quarantine.empty
        else 0,
        "issue_rows": int(len(issues)),
        "top_extreme_returns": top_cases[
            ["ticker", "date", "diagnostic_return", "classification"]
        ].to_dict("records"),
    }
    atomic_write_json(summary, OUTPUT_DIR / "anomaly_summary.json")
    atomic_write_json(summary, TABLE_DIR / "anomaly_summary.json")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "acquisition.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
