"""Standardize and audit acquired Yahoo Finance daily observations.

Raw vendor files are never zero-filled, clipped, or overwritten. The final
research-facing base intentionally leaves ``pct_return`` null. Temporary
returns are written only to a separately named diagnostic table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    atomic_write_json,
    configure_logging,
    ensure_parent,
    load_config,
    project_path,
)


LOGGER = logging.getLogger("us_equity_data.validation")

VENDOR_NUMERIC_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
]

AUDIT_COLUMNS = [
    "date",
    "ticker",
    "security_id",
    "ticker_2023",
    "provider",
    "download_id",
    "retrieved_at_utc",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "source_file",
]

BASE_COLUMNS = [
    "date",
    "security_id",
    "ticker",
    "volume",
    "adj_close",
    "pct_return",
]

DIAGNOSTIC_COLUMNS = [
    "date",
    "security_id",
    "ticker",
    "adj_close",
    "diagnostic_pct_return",
    "extreme_return_flag",
    "price_discontinuity_flag",
    "raw_adjusted_return_conflict_flag",
    "ohlc_consistency_flag",
    "zero_or_negative_adj_close_flag",
    "zero_volume_flag",
    "negative_volume_flag",
    "extreme_volume_spike_flag",
    "non_session_date_flag",
]

ISSUE_COLUMNS = [
    "issue_id",
    "security_id",
    "ticker",
    "issue_type",
    "severity",
    "status",
    "resolution",
    "affected_rows",
    "first_date",
    "last_date",
    "details",
    "suggested_action",
]

SECURITY_QUALITY_COLUMNS = [
    "security_id",
    "ticker",
    "ticker_2023",
    "company_name",
    "exchange",
    "security_type",
    "download_status",
    "raw_file",
    "first_date",
    "last_date",
    "trading_sessions",
    "valid_adj_close_sessions",
    "valid_volume_sessions",
    "missing_adj_close_observations",
    "missing_volume_observations",
    "nonpositive_adj_close_observations",
    "negative_volume_observations",
    "zero_volume_observations",
    "duplicate_rows_removed_from_base",
    "invalid_calendar_rows",
    "bad_ohlc_rows",
    "extreme_return_rows",
    "raw_adjusted_return_conflict_rows",
    "extreme_volume_spike_rows",
    "split_events",
    "dividend_events",
    "split_continuity_flags",
    "dividend_continuity_flags",
    "expected_sessions_observed_interval",
    "unobserved_internal_sessions",
    "max_unobserved_gap_sessions",
    "verified_active_missing_sessions",
    "history_status",
    "start_observation_class",
    "end_observation_class",
    "active_period_verification",
    "fewer_than_252_valid_sessions",
    "calendar_missing_rate_observed_interval",
    "row_missing_adj_close_rate",
    "row_missing_volume_rate",
    "split_volume_policy",
]

YEAR_QUALITY_COLUMNS = [
    "year",
    "securities_with_observations",
    "securities_with_valid_adj_close",
    "securities_with_valid_volume",
    "observations",
    "valid_adj_close_sessions",
    "missing_adj_close_observations",
    "missing_volume_observations",
    "extreme_return_rows",
    "raw_adjusted_return_conflict_rows",
    "extreme_volume_spike_rows",
    "split_events",
    "dividend_events",
    "expected_sessions_in_observed_intervals",
    "unobserved_sessions_within_security_intervals",
    "row_missing_adj_close_rate",
    "row_missing_volume_rate",
    "calendar_missing_rate_observed_intervals",
]


@dataclass
class FrameAudit:
    """Result of standardizing one raw vendor file."""

    audit: pd.DataFrame
    clean: pd.DataFrame
    diagnostics: pd.DataFrame
    invalid_numeric_fields: dict[str, int] = field(default_factory=dict)
    missing_vendor_fields: list[str] = field(default_factory=list)
    duplicate_rows_removed: int = 0


class IssueCollector:
    """Accumulate auditable issue groups without emitting every row."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._keys: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(
        self,
        issue_type: str,
        severity: str,
        *,
        security_id: str = "",
        ticker: str = "",
        affected_rows: int = 1,
        first_date: Any = None,
        last_date: Any = None,
        status: str = "open",
        resolution: str = "",
        details: str = "",
        suggested_action: str = "",
    ) -> None:
        key = (str(security_id), str(ticker), issue_type)
        if key in self._keys:
            row = self._keys[key]
            row["affected_rows"] += max(0, int(affected_rows))
            if pd.notna(first_date):
                row["first_date"] = min(
                    filter(pd.notna, [row["first_date"], first_date]),
                    default=first_date,
                )
            if pd.notna(last_date):
                row["last_date"] = max(
                    filter(pd.notna, [row["last_date"], last_date]),
                    default=last_date,
                )
            if details:
                existing = row["details"]
                if details not in existing:
                    row["details"] = f"{existing}; {details}"[:2000]
            return
        new_row = {
            "security_id": security_id,
            "ticker": ticker,
            "issue_type": issue_type,
            "severity": severity,
            "status": status,
            "resolution": resolution,
            "affected_rows": int(affected_rows),
            "first_date": first_date,
            "last_date": last_date,
            "details": details,
            "suggested_action": suggested_action,
        }
        self._rows.append(new_row)
        self._keys[key] = new_row

    def frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=ISSUE_COLUMNS)
        frame = pd.DataFrame(self._rows)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        frame["_severity_order"] = frame["severity"].map(severity_order).fillna(9)
        frame = frame.sort_values(
            ["_severity_order", "issue_type", "security_id", "ticker"],
            kind="stable",
        ).drop(columns="_severity_order")
        frame.insert(0, "issue_id", [f"DQ-{i:06d}" for i in range(1, len(frame) + 1)])
        return frame[ISSUE_COLUMNS]


@dataclass
class Runtime:
    config: dict[str, Any]
    raw_dir: Path
    output_dir: Path
    master: pd.DataFrame
    calendar_sessions: pd.DatetimeIndex
    calendar_date_set: set[pd.Timestamp]
    start: pd.Timestamp
    end: pd.Timestamp
    issues: IssueCollector = field(default_factory=IssueCollector)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _string(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _string(value).lower() in {"true", "1", "yes", "y"}


def _date(value: Any) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    for character in (" ", "-", "/", "(", ")"):
        text = text.replace(character, "_")
    while "__" in text:
        text = text.replace("__", "_")
    aliases = {
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "stock_split": "stock_splits",
        "stock_splits": "stock_splits",
    }
    return aliases.get(text.strip("_"), text.strip("_"))


def load_master(path: Path) -> pd.DataFrame:
    """Load optional security-master metadata, preserving missing information."""
    frame = _read_table(path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame.columns = [_normalize_name(column) for column in frame.columns]
    if "included" in frame.columns:
        frame = frame.loc[frame["included"].map(_bool)].copy()
    for column in ("listing_date", "delisting_date", "reference_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if "security_id" not in frame.columns:
        if "cik" in frame.columns:
            frame["security_id"] = frame["cik"].map(
                lambda value: f"SEC:CIK{int(float(value)):010d}"
                if pd.notna(value) and _string(value)
                else ""
            )
        else:
            frame["security_id"] = ""
    return frame


def load_calendar_sessions(calendar_name: str, start: str, end: str) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar(calendar_name)
    sessions = calendar.sessions_in_range(start, end)
    return pd.DatetimeIndex(sessions).tz_localize(None).normalize()


def standardize_frame(
    raw: pd.DataFrame,
    source_file: str = "",
    extreme_return_threshold: float = 0.50,
    extreme_volume_ratio_threshold: float = 20.0,
) -> FrameAudit:
    """Normalize one raw vendor frame while counting coercion and duplicates."""
    if raw.empty:
        return FrameAudit(
            pd.DataFrame(columns=AUDIT_COLUMNS),
            pd.DataFrame(columns=BASE_COLUMNS),
            pd.DataFrame(columns=DIAGNOSTIC_COLUMNS),
            missing_vendor_fields=VENDOR_NUMERIC_FIELDS.copy(),
        )

    frame = raw.copy()
    source_columns = list(frame.columns)
    renamed: list[Any] = []
    counts: dict[str, int] = {}
    for column in source_columns:
        name = _normalize_name(column)
        count = counts.get(name, 0) + 1
        counts[name] = count
        renamed.append(name if count == 1 else f"{name}_{count}")
    frame.columns = renamed

    if "date" not in frame.columns:
        if "index" in frame.columns:
            frame = frame.rename(columns={"index": "date"})
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index(names="date")
        else:
            raise ValueError(f"Raw file has no date column: {source_file}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    for column in ("ticker", "security_id", "ticker_2023", "provider", "download_id"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].map(_string)
    frame["source_file"] = source_file

    invalid_numeric: dict[str, int] = {}
    for column in VENDOR_NUMERIC_FIELDS:
        if column not in frame.columns:
            frame[column] = np.nan
            continue
        before = frame[column]
        numeric = pd.to_numeric(before, errors="coerce")
        invalid_numeric[column] = int(
            (before.notna() & numeric.isna() & before.astype(str).str.strip().ne("")).sum()
        )
        frame[column] = numeric
    for column in ("open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    frame = frame.loc[frame["date"].notna()].copy()
    fallback_ticker = frame["ticker_2023"].mask(
        frame["ticker_2023"].eq(""), frame["ticker"]
    ).ffill().bfill()
    frame["ticker"] = frame["ticker"].where(frame["ticker"].ne(""), fallback_ticker)
    frame["ticker_2023"] = frame["ticker_2023"].where(
        frame["ticker_2023"].ne(""), frame["ticker"]
    )
    frame["security_id"] = frame["security_id"].where(
        frame["security_id"].ne(""),
        "TICKER2023:" + frame["ticker"].fillna("UNKNOWN"),
    )
    frame = frame.sort_values(["security_id", "date"], kind="stable")

    missing_fields = [
        column
        for column in VENDOR_NUMERIC_FIELDS
        if column not in raw.columns
    ]
    for column in AUDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    audit = frame[AUDIT_COLUMNS].copy()

    key = ["security_id", "date"]
    duplicate_mask = frame.duplicated(key, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    clean = frame.loc[~duplicate_mask].copy()

    previous_adj = clean["adj_close"].shift(1)
    previous_close = clean["close"].shift(1)
    previous_volume = clean["volume"].shift(1)
    valid_return_bases = previous_adj.gt(0) & clean["adj_close"].gt(0)
    diagnostic_return = clean["adj_close"].div(previous_adj).sub(1)
    diagnostic_return = diagnostic_return.where(valid_return_bases)

    high_bad = clean["high"].lt(
        clean[["open", "close", "low"]].max(axis=1, skipna=True)
    )
    low_bad = clean["low"].gt(
        clean[["open", "close", "high"]].min(axis=1, skipna=True)
    )
    raw_gross = clean["close"].div(previous_close)
    adj_gross = clean["adj_close"].div(previous_adj)
    action_context = (
        clean["stock_splits"].gt(0).fillna(False)
        | clean["dividends"].gt(0).fillna(False)
    )
    raw_adjusted_conflict = (
        raw_gross.sub(adj_gross).abs().gt(0.01)
        & raw_gross.notna()
        & adj_gross.notna()
        & ~action_context
    )
    invalid_original_date = clean["date"].isna()
    diagnostics = pd.DataFrame(
        {
            "date": clean["date"],
            "security_id": clean["security_id"],
            "ticker": clean["ticker"],
            "adj_close": clean["adj_close"],
            "diagnostic_pct_return": diagnostic_return,
            "extreme_return_flag": diagnostic_return.abs()
            .gt(extreme_return_threshold)
            .fillna(False),
            "price_discontinuity_flag": diagnostic_return.abs()
            .gt(extreme_return_threshold)
            .fillna(False),
            "raw_adjusted_return_conflict_flag": raw_adjusted_conflict.fillna(False),
            "ohlc_consistency_flag": (high_bad | low_bad).fillna(False),
            "zero_or_negative_adj_close_flag": clean["adj_close"].le(0).fillna(False),
            "zero_volume_flag": clean["volume"].eq(0).fillna(False),
            "negative_volume_flag": clean["volume"].lt(0).fillna(False),
            "extreme_volume_spike_flag": False,
            "non_session_date_flag": invalid_original_date,
        }
    )
    if len(clean) >= 2:
        rolling_median = previous_volume.rolling(window=20, min_periods=5).median()
        diagnostics["extreme_volume_spike_flag"] = (
            rolling_median.gt(0)
            & clean["volume"].gt(rolling_median * extreme_volume_ratio_threshold)
        ).fillna(False)

    return FrameAudit(
        audit=audit,
        clean=clean,
        diagnostics=diagnostics[DIAGNOSTIC_COLUMNS],
        invalid_numeric_fields={
            key: value for key, value in invalid_numeric.items() if value
        },
        missing_vendor_fields=missing_fields,
        duplicate_rows_removed=duplicate_count,
    )


def _coverage_for_interval(
    observed_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    all_sessions: np.ndarray,
) -> tuple[int, np.ndarray]:
    if pd.isna(start) or pd.isna(end) or start > end:
        return 0, np.array([], dtype=int)
    left = int(np.searchsorted(all_sessions, np.datetime64(start.normalize()), "left"))
    right = int(np.searchsorted(all_sessions, np.datetime64(end.normalize()), "right"))
    interval = all_sessions[left:right]
    observed = np.array(
        sorted({np.datetime64(value.normalize()) for value in observed_dates}),
        dtype="datetime64[ns]",
    )
    if len(interval) == 0 or len(observed) == 0:
        return len(interval), np.arange(len(interval), dtype=int)
    unobserved = np.setdiff1d(interval, observed, assume_unique=True)
    if len(unobserved) == 0:
        return len(interval), np.array([], dtype=int)
    indexes = np.searchsorted(interval, unobserved)
    return len(interval), indexes


def _longest_consecutive(indexes: np.ndarray) -> int:
    if len(indexes) == 0:
        return 0
    breaks = np.flatnonzero(np.diff(indexes) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(indexes) - 1]
    return int(np.max(ends - starts + 1))


def _metadata_for(
    frame: pd.DataFrame, master_by_id: dict[str, pd.Series]
) -> tuple[pd.Series | None, dict[str, Any]]:
    security_id = _string(frame["security_id"].iloc[0])
    metadata = master_by_id.get(security_id)
    if metadata is None:
        return None, {}
    return metadata, {
        "company_name": _string(metadata.get("company_name", metadata.get("name", ""))),
        "exchange": _string(metadata.get("exchange", "")),
        "security_type": _string(metadata.get("security_type", "")),
        "listing_date": _date(metadata.get("listing_date")),
        "delisting_date": _date(metadata.get("delisting_date")),
        "reference_date_verified": _bool(metadata.get("reference_date_verified", False)),
    }


def _corporate_action_checks(
    clean: pd.DataFrame,
    ticker: str,
    security_id: str,
    split_tolerance: float,
    dividend_tolerance: float,
    issues: IssueCollector,
) -> tuple[int, int]:
    """Return counts of split and dividend continuity flags."""
    if len(clean) < 2:
        return 0, 0
    previous = clean.shift(1)
    raw_ratio = previous["close"] / clean["close"]
    adj_gross = clean["adj_close"] / previous["adj_close"]

    split_mask = clean["stock_splits"].gt(0).fillna(False)
    dividend_mask = clean["dividends"].gt(0).fillna(False)
    combined_action = split_mask & dividend_mask
    simple_split = split_mask & ~combined_action
    split_flags = mask = pd.Series(False, index=clean.index)
    if simple_split.any():
        # The raw pre/post-close ratio combines the split factor and the
        implied = raw_ratio.loc[simple_split] * adj_gross.loc[simple_split]
        provider = _string(clean["provider"].iloc[0]).lower()
        if "yahoo" in provider:
            # Yahoo's nominal Close is back-adjusted across splits even when
            # auto_adjust=False. A continuous raw/Adj Close pair therefore
            # implies unity, while the split multiplier remains metadata.
            expected = pd.Series(1.0, index=implied.index)
        else:
            # Unadjusted close series, such as the secondary audit source,
            # imply the vendor event multiplier directly.
            expected = clean.loc[simple_split, "stock_splits"]
        relative_error = (
            implied.sub(expected).abs() / expected.abs()
        )
        split_flags.loc[simple_split] = relative_error.gt(split_tolerance).fillna(True)
        bad = split_flags
        if bad.any():
            issues.add(
                "split_adjustment_continuity",
                "medium",
                security_id=security_id,
                ticker=ticker,
                affected_rows=int(bad.sum()),
                first_date=clean.loc[bad, "date"].min(),
                last_date=clean.loc[bad, "date"].max(),
                details=(
                    "Raw-close and adjusted-close continuity differs from the "
                    "expected convention for this vendor beyond tolerance."
                ),
                suggested_action="Review the event, raw prices, and adjusted-price continuity.",
            )

        volume_ratio = clean["volume"] / previous["volume"]
        event_volumes = volume_ratio.loc[simple_split]
        ratios = clean.loc[simple_split, "stock_splits"]
        level_change = (
            event_volumes.div(ratios).sub(1).abs().lt(0.20)
            & previous.loc[simple_split, "volume"].gt(0)
            & clean.loc[simple_split, "volume"].gt(0)
        )
        if level_change.any():
            issues.add(
                "split_volume_level_change_observed",
                "info",
                security_id=security_id,
                ticker=ticker,
                affected_rows=int(level_change.sum()),
                first_date=clean.loc[simple_split, "date"].loc[level_change].min(),
                last_date=clean.loc[simple_split, "date"].loc[level_change].max(),
                status="documented",
                resolution=(
                    "Volume is retained in reported share units; it is not assumed "
                    "to be split-normalized."
                ),
                details=(
                    "Reported volume changed by approximately the split factor on "
                    "an event date. This is descriptive, not proof of an error."
                ),
                suggested_action="Normalize only inside a later explicitly defined transform.",
            )

    if combined_action.any():
        issues.add(
            "simultaneous_split_and_distribution",
            "info",
            security_id=security_id,
            ticker=ticker,
            affected_rows=int(combined_action.sum()),
            first_date=clean.loc[combined_action, "date"].min(),
            last_date=clean.loc[combined_action, "date"].max(),
            status="documented",
            resolution=(
                "Combined actions are excluded from simple split and dividend "
                "continuity tests and retained for event-level review."
            ),
            details=(
                "A split and cash/distribution event occur on the same date; "
                "a single-action continuity formula is not applicable."
            ),
            suggested_action=(
                "Reconcile event type and combined adjustment before return use."
            ),
        )

    simple_dividend = dividend_mask & ~combined_action
    dividend_flags = pd.Series(False, index=clean.index)
    if simple_dividend.any():
        raw_total_return = (
            clean["close"] + clean["dividends"].where(simple_dividend, 0.0)
        ).div(previous["close"]).sub(1)
        dividend_error = adj_gross.sub(1).sub(raw_total_return).abs()
        dividend_flags.loc[simple_dividend] = (
            dividend_error.loc[simple_dividend]
            .gt(dividend_tolerance)
            .fillna(True)
        )
        if dividend_flags.any():
            issues.add(
                "dividend_adjustment_continuity",
                "medium",
                security_id=security_id,
                ticker=ticker,
                affected_rows=int(dividend_flags.sum()),
                first_date=clean.loc[dividend_flags, "date"].min(),
                last_date=clean.loc[dividend_flags, "date"].max(),
                details=(
                    "Adjusted-close total return differs from raw close plus the "
                    "reported dividend beyond tolerance."
                ),
                suggested_action="Review vendor adjustment around the dividend event.",
            )

    return int(split_flags.sum()), int(dividend_flags.sum())


def _quality_for_frame(
    audit: FrameAudit,
    runtime: Runtime,
    source_path: Path,
    master_by_id: dict[str, pd.Series],
    manifest_status: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute one security's quality record and per-year components."""
    clean = audit.clean
    if clean.empty:
        return {}, []
    security_id = _string(clean["security_id"].iloc[0])
    ticker = _string(clean["ticker"].iloc[0])
    metadata, meta = _metadata_for(clean, master_by_id)
    first_date = clean["date"].min().normalize()
    last_date = clean["date"].max().normalize()
    observed_dates = pd.DatetimeIndex(clean["date"].dt.normalize().unique())
    calendar_observed = observed_dates.intersection(runtime.calendar_sessions)
    non_session_count = int(len(clean) - len(calendar_observed))
    if non_session_count:
        runtime.issues.add(
            "non_session_observation_date",
            "high",
            security_id=security_id,
            ticker=ticker,
            affected_rows=non_session_count,
            first_date=first_date,
            last_date=last_date,
            details="Vendor rows fall on dates outside the configured XNYS calendar.",
            suggested_action="Review timestamp/timezone conventions before research use.",
        )

    expected_count, unobserved_indexes = _coverage_for_interval(
        calendar_observed,
        first_date,
        last_date,
        runtime.calendar_sessions.values,
    )
    unobserved_count = int(len(unobserved_indexes))
    max_gap = _longest_consecutive(unobserved_indexes)
    maximum_gap = int(runtime.config["validation"]["max_unexpected_gap_sessions"])
    if unobserved_count:
        severity = "high" if max_gap > maximum_gap else "medium"
        runtime.issues.add(
            "unobserved_sessions_inside_history",
            severity,
            security_id=security_id,
            ticker=ticker,
            affected_rows=unobserved_count,
            first_date=runtime.calendar_sessions[unobserved_indexes[0]],
            last_date=runtime.calendar_sessions[unobserved_indexes[-1]],
            details=(
                "Expected exchange sessions are absent between the first and last "
                f"observed dates; longest contiguous run is {max_gap} sessions. "
                "Cause is not inferred."
            ),
            suggested_action=(
                "Cross-check exchange notices and listing status; do not fill zero."
            ),
        )

    listing_date = meta.get("listing_date", pd.NaT)
    delisting_date = meta.get("delisting_date", pd.NaT)
    start_class = "unresolved_before_first_observation"
    if pd.notna(listing_date) and first_date >= listing_date:
        start_class = "pre_listing_structurally_unobserved"
    end_class = "unknown_after_last_observation"
    if pd.notna(delisting_date):
        if abs(int(runtime.calendar_sessions.searchsorted(last_date)) - int(runtime.calendar_sessions.searchsorted(delisting_date))) <= maximum_gap:
            end_class = "post_delisting_structurally_unobserved"
        elif last_date < delisting_date:
            end_class = "vendor_history_ends_before_verified_delisting"
            runtime.issues.add(
                "history_ends_before_verified_delisting",
                "high",
                security_id=security_id,
                ticker=ticker,
                affected_rows=0,
                first_date=last_date,
                last_date=delisting_date,
                details=(
                    "The last observation precedes the verified delisting date by "
                    "more than the configured gap-review threshold."
                ),
                suggested_action="Retry historical acquisition and review delisting metadata.",
            )
        else:
            end_class = "observations_after_verified_delisting"
            runtime.issues.add(
                "observations_after_verified_delisting",
                "high",
                security_id=security_id,
                ticker=ticker,
                affected_rows=0,
                first_date=delisting_date,
                last_date=last_date,
                details="Observations extend beyond the security master's delisting date.",
                suggested_action="Resolve the identifier, ticker reuse, or master-date error.",
            )

    if pd.notna(listing_date):
        expected_start = max(listing_date.normalize(), runtime.start)
        if first_date > expected_start:
            _, missing_before_first = _coverage_for_interval(
                calendar_observed,
                expected_start,
                first_date - pd.Timedelta(days=1),
                runtime.calendar_sessions.values,
            )
            if len(missing_before_first):
                runtime.issues.add(
                    "history_starts_after_verified_listing",
                    "high",
                    security_id=security_id,
                    ticker=ticker,
                    affected_rows=len(missing_before_first),
                    first_date=expected_start,
                    last_date=first_date - pd.Timedelta(days=1),
                    details=(
                        "Vendor observations begin after the verified listing date "
                        "within the requested research window."
                    ),
                    suggested_action="Retry acquisition and verify ticker mapping.",
                )

    active_verification = "unverified"
    verified_active_missing = np.nan
    if pd.notna(listing_date) or pd.notna(delisting_date):
        active_start = (
            max(listing_date.normalize(), runtime.start)
            if pd.notna(listing_date)
            else first_date
        )
        active_end = (
            min(delisting_date.normalize(), runtime.end)
            if pd.notna(delisting_date)
            else last_date
        )
        if active_start <= active_end:
            active_verification = (
                "verified_listing_and_delisting_dates"
                if pd.notna(listing_date) and pd.notna(delisting_date)
                else (
                    "listing_date_verified_end_unknown"
                    if pd.notna(listing_date)
                    else "delisting_date_verified_start_unknown"
                )
            )
        else:
            active_verification = "inconsistent_verified_active_period"
            runtime.issues.add(
                "inconsistent_listing_dates",
                "high",
                security_id=security_id,
                ticker=ticker,
                affected_rows=0,
                first_date=listing_date if pd.notna(listing_date) else pd.NaT,
                last_date=delisting_date if pd.notna(delisting_date) else pd.NaT,
                details="Verified listing and delisting dates do not form a valid active interval.",
                suggested_action="Resolve source date and security-identity conflict.",
            )
        if active_verification.startswith("verified_") or active_verification.startswith("listing_") or active_verification.startswith("delisting_"):
            active_start = max(active_start, runtime.start)
            active_end = min(active_end, runtime.end)
        _, verified_missing = _coverage_for_interval(
            calendar_observed,
            active_start,
            active_end,
            runtime.calendar_sessions.values,
        )
        verified_active_missing = len(verified_missing)

    final_session = runtime.calendar_sessions[-1]
    first_session = runtime.calendar_sessions[0]
    if first_date > first_session and last_date < final_session:
        history_status = "partial_both_ends"
    elif first_date > first_session:
        history_status = "starts_after_window"
    elif last_date < final_session:
        history_status = "ends_before_window"
    else:
        history_status = "covers_window_ends"

    on_calendar = clean["date"].dt.normalize().isin(runtime.calendar_sessions)
    valid_adj = clean["adj_close"].gt(0) & on_calendar
    valid_volume = clean["volume"].gt(0) & on_calendar
    missing_adj = clean["adj_close"].isna()
    missing_volume = clean["volume"].isna()
    nonpositive_adj = clean["adj_close"].le(0)
    negative_volume = clean["volume"].lt(0)
    zero_volume = clean["volume"].eq(0)
    bad_ohlc = audit.diagnostics["ohlc_consistency_flag"]
    extreme_return = audit.diagnostics["extreme_return_flag"]
    raw_adjusted_conflicts = audit.diagnostics[
        "raw_adjusted_return_conflict_flag"
    ]
    extreme_volume = audit.diagnostics["extreme_volume_spike_flag"]
    split_events = clean["stock_splits"].gt(0)
    dividend_events = clean["dividends"].gt(0)

    split_flags, dividend_flags = _corporate_action_checks(
        clean,
        ticker,
        security_id,
        float(runtime.config["validation"]["split_relative_tolerance"]),
        float(runtime.config["validation"]["dividend_adjustment_tolerance"]),
        runtime.issues,
    )

    checks = [
        (
            "missing_adj_close",
            "high" if missing_adj.mean() > 0.05 else "medium",
            int(missing_adj.sum()),
            "Adjusted close is null in acquired vendor rows.",
            "Investigate provider history; never fill with zero.",
        ),
        (
            "nonpositive_adj_close",
            "high",
            int(nonpositive_adj.sum()),
            "Adjusted close is zero or negative.",
            "Review the raw observation and corporate-action context.",
        ),
        (
            "missing_volume",
            "medium",
            int(missing_volume.sum()),
            "Share volume is null in acquired vendor rows.",
            "Retain as missing and investigate vendor coverage.",
        ),
        (
            "negative_volume",
            "high",
            int(negative_volume.sum()),
            "Share volume is negative.",
            "Treat as invalid pending review; do not silently replace.",
        ),
        (
            "zero_volume",
            "low",
            int(zero_volume.sum()),
            "Share volume is reported as zero.",
            "Retain the observation; zero can be legitimate for illiquid names.",
        ),
        (
            "ohlc_consistency",
            "high",
            int(bad_ohlc.sum()),
            "High/low do not bound open/close in the vendor row.",
            "Review the observation and any action event.",
        ),
        (
            "extreme_diagnostic_return",
            "medium",
            int(extreme_return.sum()),
            "Absolute temporary adjusted return exceeds the review threshold.",
            "Review as market shock or data issue; do not clip automatically.",
        ),
        (
            "raw_adjusted_return_conflict",
            "medium",
            int(raw_adjusted_conflicts.sum()),
            "Raw and adjusted close imply different same-day returns without a reported action.",
            "Audit the vendor adjustment factor and the raw observation.",
        ),
        (
            "extreme_diagnostic_volume_spike",
            "low",
            int(extreme_volume.sum()),
            "Volume exceeds 20 times a preceding rolling median in this diagnostic check.",
            "Review news, listing changes, and action events; no automatic removal.",
        ),
    ]
    for issue_type, severity, count, details, action in checks:
        if count:
            mask: pd.Series = {
                "missing_adj_close": missing_adj,
                "nonpositive_adj_close": nonpositive_adj,
                "missing_volume": missing_volume,
                "negative_volume": negative_volume,
                "zero_volume": zero_volume,
                "ohlc_consistency": bad_ohlc,
                "extreme_diagnostic_return": extreme_return,
                "raw_adjusted_return_conflict": raw_adjusted_conflicts,
                "extreme_diagnostic_volume_spike": extreme_volume,
            }[issue_type]
            runtime.issues.add(
                issue_type,
                severity,
                security_id=security_id,
                ticker=ticker,
                affected_rows=count,
                first_date=clean.loc[mask, "date"].min(),
                last_date=clean.loc[mask, "date"].max(),
                details=details,
                suggested_action=action,
            )

    if audit.duplicate_rows_removed:
        runtime.issues.add(
            "duplicate_security_date_rows",
            "high",
            security_id=security_id,
            ticker=ticker,
            affected_rows=audit.duplicate_rows_removed,
            first_date=first_date,
            last_date=last_date,
            resolution="First vendor row retained in base; raw duplicate retained in raw file.",
            details="Duplicate security-date keys were detected.",
            suggested_action="Review vendor file provenance and acquisition retries.",
        )
    if audit.missing_vendor_fields:
        runtime.issues.add(
            "missing_vendor_fields",
            "high",
            security_id=security_id,
            ticker=ticker,
            affected_rows=len(audit.missing_vendor_fields),
            details="Missing required vendor columns: " + ", ".join(audit.missing_vendor_fields),
            suggested_action="Verify download parameters before treating the file as complete.",
        )
    if audit.invalid_numeric_fields:
        runtime.issues.add(
            "invalid_numeric_vendor_values",
            "medium",
            security_id=security_id,
            ticker=ticker,
            affected_rows=sum(audit.invalid_numeric_fields.values()),
            details=json.dumps(audit.invalid_numeric_fields, sort_keys=True),
            suggested_action="Inspect raw strings; validation coercion did not alter raw files.",
        )

    minimum = int(runtime.config["validation"]["minimum_valid_sessions"])
    quality = {
        "security_id": security_id,
        "ticker": ticker,
        "ticker_2023": _string(clean["ticker_2023"].iloc[0]) or ticker,
        "company_name": meta.get("company_name", ""),
        "exchange": meta.get("exchange", ""),
        "security_type": meta.get("security_type", ""),
        "download_status": manifest_status.get(security_id, "raw_present_not_in_manifest"),
        "raw_file": str(source_path),
        "first_date": first_date,
        "last_date": last_date,
        "trading_sessions": int(on_calendar.sum()),
        "valid_adj_close_sessions": int(valid_adj.sum()),
        "valid_volume_sessions": int(valid_volume.sum()),
        "missing_adj_close_observations": int(missing_adj.sum()),
        "missing_volume_observations": int(missing_volume.sum()),
        "nonpositive_adj_close_observations": int(nonpositive_adj.sum()),
        "negative_volume_observations": int(negative_volume.sum()),
        "zero_volume_observations": int(zero_volume.sum()),
        "duplicate_rows_removed_from_base": audit.duplicate_rows_removed,
        "invalid_calendar_rows": non_session_count,
        "bad_ohlc_rows": int(bad_ohlc.sum()),
        "extreme_return_rows": int(extreme_return.sum()),
        "raw_adjusted_return_conflict_rows": int(
            raw_adjusted_conflicts.sum()
        ),
        "extreme_volume_spike_rows": int(extreme_volume.sum()),
        "split_events": int(split_events.sum()),
        "dividend_events": int(dividend_events.sum()),
        "split_continuity_flags": split_flags,
        "dividend_continuity_flags": dividend_flags,
        "expected_sessions_observed_interval": expected_count,
        "unobserved_internal_sessions": unobserved_count,
        "max_unobserved_gap_sessions": max_gap,
        "verified_active_missing_sessions": verified_active_missing,
        "history_status": history_status,
        "start_observation_class": start_class,
        "end_observation_class": end_class,
        "active_period_verification": active_verification,
        "fewer_than_252_valid_sessions": bool(valid_adj.sum() < minimum),
        "calendar_missing_rate_observed_interval": (
            unobserved_count / expected_count if expected_count else np.nan
        ),
        "row_missing_adj_close_rate": float(missing_adj.mean()),
        "row_missing_volume_rate": float(missing_volume.mean()),
        "split_volume_policy": (
            "Reported share volume retained without normalization; compare "
            "pre/post-split volume only after an explicit later transformation."
        ),
    }

    year_rows: list[dict[str, Any]] = []
    for year in range(runtime.start.year, runtime.end.year + 1):
        year_start = max(pd.Timestamp(year=year, month=1, day=1), first_date)
        year_end = min(pd.Timestamp(year=year, month=12, day=31), last_date)
        if year_start > year_end:
            continue
        in_year = clean["date"].dt.year.eq(year)
        year_observed = pd.DatetimeIndex(
            clean.loc[in_year, "date"].dt.normalize().unique()
        ).intersection(runtime.calendar_sessions)
        expected_year, missing_year = _coverage_for_interval(
            year_observed,
            year_start,
            year_end,
            runtime.calendar_sessions.values,
        )
        year_rows.append(
            {
                "year": year,
                "securities_with_observations": int(in_year.any()),
                "securities_with_valid_adj_close": int(
                    (in_year & valid_adj).any()
                ),
                "securities_with_valid_volume": int(
                    (in_year & valid_volume).any()
                ),
                "observations": int(in_year.sum()),
                "valid_adj_close_sessions": int((in_year & valid_adj).sum()),
                "missing_adj_close_observations": int((in_year & missing_adj).sum()),
                "missing_volume_observations": int((in_year & missing_volume).sum()),
                "extreme_return_rows": int((in_year & extreme_return).sum()),
                "raw_adjusted_return_conflict_rows": int(
                    (in_year & raw_adjusted_conflicts).sum()
                ),
                "extreme_volume_spike_rows": int((in_year & extreme_volume).sum()),
                "split_events": int((in_year & split_events).sum()),
                "dividend_events": int((in_year & dividend_events).sum()),
                "expected_sessions_in_observed_intervals": expected_year,
                "unobserved_sessions_within_security_intervals": len(missing_year),
            }
        )
    return quality, year_rows


def _schema_for(columns: Iterable[str]) -> pa.Schema:
    fields: list[pa.Field] = []
    for column in columns:
        if column == "date" or column.endswith("_date"):
            fields.append(pa.field(column, pa.timestamp("ns")))
        elif column.endswith("_flag") or column.startswith("fewer_than_"):
            fields.append(pa.field(column, pa.bool_()))
        elif column in {
            "volume",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "dividends",
            "stock_splits",
            "pct_return",
            "diagnostic_pct_return",
        }:
            fields.append(pa.field(column, pa.float64()))
        else:
            fields.append(pa.field(column, pa.string()))
    return pa.schema(fields)


class ParquetSink:
    """Incremental Parquet writer for large per-security streams."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = ensure_parent(path)
        self.schema = _schema_for(columns)
        self.columns = columns
        self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        payload = frame[self.columns].copy()
        for field in self.schema:
            if pa.types.is_timestamp(field.type):
                payload[field.name] = pd.to_datetime(payload[field.name], errors="coerce")
            elif pa.types.is_string(field.type):
                payload[field.name] = payload[field.name].map(_string)
            elif pa.types.is_boolean(field.type):
                payload[field.name] = payload[field.name].map(_bool)
            elif pa.types.is_floating(field.type):
                payload[field.name] = pd.to_numeric(
                    payload[field.name], errors="coerce"
                ).astype("float64")
        table = pa.Table.from_pandas(payload, schema=self.schema, preserve_index=False)
        self.writer.write_table(table)
        self.rows += len(payload)

    def close(self) -> None:
        self.writer.close()

    def empty_close(self) -> None:
        self.writer.close()
        empty = pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in self.schema],
            schema=self.schema,
        )
        pq.write_table(empty, self.path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=target.parent, delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_status(
    metadata_dir: Path,
    manifest_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    statuses: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    paths = (
        project_path(manifest_path)
        if manifest_path
        else metadata_dir / "download_manifest.csv",
        project_path(failures_path)
        if failures_path
        else metadata_dir / "download_failures.csv",
    )
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            if not frame.empty:
                frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not manifest.empty and "security_id" in manifest.columns:
        # The failure register is normally a filtered view of the manifest.
        # Retain one row per security to avoid duplicate issue counts.
        manifest = manifest.drop_duplicates("security_id", keep="last")
        for _, row in manifest.iterrows():
            security_id = _string(row.get("security_id"))
            if security_id:
                statuses[security_id] = _string(
                    row.get("download_status", "unknown")
                )
    return statuses, manifest


def _acquisition_issues(
    manifest: pd.DataFrame,
    master: pd.DataFrame,
    discovered_ids: set[str],
    issues: IssueCollector,
) -> None:
    if manifest.empty:
        return
    for _, row in manifest.iterrows():
        security_id = _string(row.get("security_id"))
        ticker = _string(row.get("yahoo_ticker", row.get("ticker", "")))
        status = _string(row.get("download_status", "")).lower()
        if (
            status
            and status not in {"success", "resumed", "partial"}
            and security_id not in discovered_ids
        ):
            issues.add(
                "download_unavailable",
                "high",
                security_id=security_id,
                ticker=ticker,
                status="open",
                details=(
                    f"Acquisition status={status or 'unknown'}; "
                    f"error={_string(row.get('error', 'unknown'))}"
                ),
                suggested_action=(
                    "Retry through the documented acquisition process and retain "
                    "the failure in coverage reporting."
                ),
            )
        elif status == "partial":
            issues.add(
                "partial_vendor_history",
                "medium",
                security_id=security_id,
                ticker=ticker,
                affected_rows=0,
                details=_string(row.get("history_status", "partial history")),
                suggested_action="Compare first/last dates with verified listing status.",
            )

    if "included" in master.columns:
        included_ids = set(master.loc[master["included"].map(_bool), "security_id"].map(_string))
        attempted_ids = set(manifest["security_id"].map(_string)) if "security_id" in manifest else set()
        unattempted = included_ids - attempted_ids
        if unattempted:
            issues.add(
                "universe_security_not_attempted",
                "high",
                affected_rows=len(unattempted),
                details=(
                    f"{len(unattempted)} included security identifiers have no "
                    "acquisition-manifest row."
                ),
                suggested_action="Run acquisition to completion or report unexecuted coverage.",
            )


def identity_audit(
    master: pd.DataFrame, issues: IssueCollector
) -> pd.DataFrame:
    """Audit ticker reuse and unresolved security identifiers in the master."""
    rows: list[dict[str, Any]] = []
    if master.empty:
        return pd.DataFrame(
            columns=[
                "key_type",
                "key",
                "security_ids",
                "tickers",
                "affected_securities",
                "finding",
                "severity",
            ]
        )

    for column in ("yahoo_ticker", "ticker_2023", "ticker"):
        if column not in master.columns:
            continue
        table = master.loc[
            master[column].map(_string).ne(""),
            ["security_id", column],
        ].copy()
        table["security_id"] = table["security_id"].map(_string)
        table[column] = table[column].map(_string).str.upper()
        groups = table.groupby(column, dropna=False)["security_id"].agg(
            lambda values: sorted(set(value for value in values if value))
        )
        for ticker, security_ids in groups.items():
            if len(security_ids) <= 1:
                continue
            rows.append(
                {
                    "key_type": column,
                    "key": ticker,
                    "security_ids": ";".join(security_ids),
                    "tickers": ticker,
                    "affected_securities": len(security_ids),
                    "finding": "ticker_mapped_to_multiple_security_ids",
                    "severity": "high",
                }
            )
            issues.add(
                "ticker_reuse_or_ambiguous_mapping",
                "high",
                ticker=ticker,
                affected_rows=len(security_ids),
                details=(
                    f"{column}={ticker} maps to {len(security_ids)} security IDs: "
                    + ", ".join(security_ids)
                ),
                suggested_action=(
                    "Resolve temporal ticker reuse and preserve distinct security identities."
                ),
            )

    missing_id = master["security_id"].map(_string).eq("")
    if missing_id.any():
        rows.append(
            {
                "key_type": "security_id",
                "key": "<missing>",
                "security_ids": "",
                "tickers": "",
                "affected_securities": int(missing_id.sum()),
                "finding": "missing_security_id",
                "severity": "high",
            }
        )
        issues.add(
            "missing_security_id",
            "high",
            affected_rows=int(missing_id.sum()),
            details="Included security-master rows lack a security ID.",
            suggested_action="Resolve identity before acquisition or analysis.",
        )

    supported_id = master["security_id"].map(_string).str.startswith(
        ("SEC:", "TICKER2023:")
    )
    unsupported = master["security_id"].map(_string).ne("") & ~supported_id
    if unsupported.any():
        rows.append(
            {
                "key_type": "security_id",
                "key": "<unsupported_prefix>",
                "security_ids": ";".join(
                    sorted(master.loc[unsupported, "security_id"].map(_string))[:50]
                ),
                "tickers": "",
                "affected_securities": int(unsupported.sum()),
                "finding": "unsupported_security_identifier",
                "severity": "medium",
            }
        )
        issues.add(
            "unsupported_security_identifier",
            "medium",
            affected_rows=int(unsupported.sum()),
            details="Security IDs do not use a documented SEC or fallback prefix.",
            suggested_action="Map identifiers to the shared schema or retain an explicit issue.",
        )

    if "reference_date_verified" in master.columns:
        unverified = ~master["reference_date_verified"].map(_bool)
        if unverified.any():
            rows.append(
                {
                    "key_type": "reference_date_verified",
                    "key": "<unverified>",
                    "security_ids": "",
                    "tickers": "",
                    "affected_securities": int(unverified.sum()),
                    "finding": "reference_date_unverified",
                    "severity": "high",
                }
            )
            issues.add(
                "reference_date_not_verified",
                "high",
                affected_rows=int(unverified.sum()),
                details=(
                    "Included securities lack a verified 2023-01-03 eligibility "
                    "decision in the available source fields."
                ),
                suggested_action=(
                    "Do not present these rows as a complete reference-date universe."
                ),
            )
    return pd.DataFrame(rows)


def validate(
    config: dict[str, Any],
    raw_dir: str | Path,
    output_dir: str | Path,
    master_path: str | Path,
    metadata_dir: str | Path | None = None,
    data_mode: str = "live_or_cli_input",
    manifest_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full validation pipeline and return a compact run summary."""
    start = pd.Timestamp(config["project"]["start_date"]).normalize()
    end = pd.Timestamp(config["project"]["end_date_exclusive"]).normalize() - pd.Timedelta(days=1)
    calendar_sessions = load_calendar_sessions(
        config["validation"]["us_calendar"],
        start.date().isoformat(),
        end.date().isoformat(),
    )
    master = load_master(project_path(master_path))
    runtime = Runtime(
        config=config,
        raw_dir=project_path(raw_dir),
        output_dir=ensure_parent(project_path(output_dir) / "placeholder"),
        master=master,
        calendar_sessions=calendar_sessions,
        calendar_date_set=set(calendar_sessions),
        start=start,
        end=end,
    )
    runtime.output_dir = project_path(output_dir)
    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = project_path(metadata_dir) if metadata_dir else project_path(config["paths"]["metadata_dir"])
    statuses, manifest = _manifest_status(
        metadata_path,
        manifest_path=manifest_path,
        failures_path=failures_path,
    )
    identity_findings = identity_audit(master, runtime.issues)

    master_by_id: dict[str, pd.Series] = {}
    if not master.empty:
        for _, row in master.iterrows():
            security_id = _string(row.get("security_id"))
            if security_id:
                master_by_id[security_id] = row

    base_sink = ParquetSink(runtime.output_dir / "base.parquet", BASE_COLUMNS)
    audit_sink = ParquetSink(
        runtime.output_dir / "standardized_raw.parquet", AUDIT_COLUMNS
    )
    diagnostic_sink = ParquetSink(
        runtime.output_dir / "diagnostic_returns.parquet", DIAGNOSTIC_COLUMNS
    )
    security_rows: list[dict[str, Any]] = []
    year_accumulator: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    discovered_ids: set[str] = set()
    processed_paths = sorted(runtime.raw_dir.rglob("*.parquet")) if runtime.raw_dir.exists() else []

    for path in processed_paths:
        LOGGER.info("Validating %s", path)
        raw = pd.read_parquet(path)
        relative_source = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
        audit = standardize_frame(
            raw,
            relative_source,
            extreme_return_threshold=float(
                config["validation"]["extreme_return_abs_threshold"]
            ),
            extreme_volume_ratio_threshold=float(
                config["validation"]["extreme_volume_ratio_threshold"]
            ),
        )
        if audit.clean.empty:
            runtime.issues.add(
                "empty_raw_file",
                "high",
                details=f"No usable dated observations in {relative_source}.",
                suggested_action="Retry download and inspect manifest failure metadata.",
            )
            continue
        security_id = _string(audit.clean["security_id"].iloc[0])
        discovered_ids.add(security_id)
        if not audit.diagnostics.empty:
            audit.diagnostics["non_session_date_flag"] = (
                ~audit.diagnostics["date"]
                .dt.normalize()
                .isin(runtime.calendar_sessions)
            )
        audit_sink.write(audit.audit)
        # The research-facing base contains valid standardized observations.
        # Invalid or placeholder values remain in raw/audit and issue outputs.
        valid_base_rows = (
            audit.clean["adj_close"].gt(0)
            & audit.clean["volume"].notna()
            & audit.clean["volume"].ge(0)
        )
        base = audit.clean.loc[valid_base_rows].assign(pct_return=np.nan)[BASE_COLUMNS].copy()
        base["ticker"] = audit.clean.loc[valid_base_rows, "ticker_2023"].where(
            audit.clean.loc[valid_base_rows, "ticker_2023"].ne(""),
            audit.clean.loc[valid_base_rows, "ticker"],
        )
        base_sink.write(base)
        diagnostic_sink.write(audit.diagnostics)
        quality, year_rows = _quality_for_frame(
            audit, runtime, path, master_by_id, statuses
        )
        if quality:
            security_rows.append(quality)
        for row in year_rows:
            for key, value in row.items():
                year_accumulator[int(row["year"])][key] += value

    base_sink.close()
    audit_sink.close()
    diagnostic_sink.close()

    attempted_without_raw: set[str] = set()
    if not manifest.empty:
        attempted_without_raw = {
            _string(value)
            for value in manifest.get("security_id", pd.Series(dtype=object))
            if _string(value) and _string(value) not in discovered_ids
        }
    present_ids = {row["security_id"] for row in security_rows}
    universe_ids = set(master_by_id) if master_by_id else set()
    extra_ids = (universe_ids | attempted_without_raw) - present_ids
    for security_id in sorted(extra_ids):
        metadata = master_by_id.get(security_id, pd.Series(dtype=object))
        status = statuses.get(security_id, "not_attempted")
        security_rows.append(
            {
                "security_id": security_id,
                "ticker": _string(metadata.get("yahoo_ticker", metadata.get("ticker_2023", ""))),
                "ticker_2023": _string(metadata.get("ticker_2023", metadata.get("ticker", ""))),
                "company_name": _string(metadata.get("company_name", metadata.get("name", ""))),
                "exchange": _string(metadata.get("exchange", "")),
                "security_type": _string(metadata.get("security_type", "")),
                "download_status": status,
                "raw_file": "",
                "first_date": pd.NaT,
                "last_date": pd.NaT,
                "trading_sessions": 0,
                "valid_adj_close_sessions": 0,
                "valid_volume_sessions": 0,
                "missing_adj_close_observations": 0,
                "missing_volume_observations": 0,
                "nonpositive_adj_close_observations": 0,
                "negative_volume_observations": 0,
                "zero_volume_observations": 0,
                "duplicate_rows_removed_from_base": 0,
                "invalid_calendar_rows": 0,
                "bad_ohlc_rows": 0,
                "extreme_return_rows": 0,
                "raw_adjusted_return_conflict_rows": 0,
                "extreme_volume_spike_rows": 0,
                "split_events": 0,
                "dividend_events": 0,
                "split_continuity_flags": 0,
                "dividend_continuity_flags": 0,
                "expected_sessions_observed_interval": 0,
                "unobserved_internal_sessions": 0,
                "max_unobserved_gap_sessions": 0,
                "verified_active_missing_sessions": np.nan,
                "history_status": "no_observations",
                "start_observation_class": "not_observed",
                "end_observation_class": "not_observed",
                "active_period_verification": "unverified",
                "fewer_than_252_valid_sessions": True,
                "calendar_missing_rate_observed_interval": np.nan,
                "row_missing_adj_close_rate": np.nan,
                "row_missing_volume_rate": np.nan,
                "split_volume_policy": (
                    "Reported share volume retained without normalization; compare "
                    "pre/post-split volume only after an explicit later transformation."
                ),
            }
        )

    security_quality = pd.DataFrame(security_rows, columns=SECURITY_QUALITY_COLUMNS)
    if not security_quality.empty:
        security_quality = security_quality.sort_values(
            ["security_id", "ticker"], kind="stable"
        )
    _acquisition_issues(manifest, master, discovered_ids, runtime.issues)

    year_rows: list[dict[str, Any]] = []
    for year in range(start.year, end.year + 1):
        values = year_accumulator.get(year, {})
        observations = values.get("observations", 0)
        expected = values.get("expected_sessions_in_observed_intervals", 0)
        missing = values.get("unobserved_sessions_within_security_intervals", 0)
        year_rows.append(
            {
                "year": year,
                "securities_with_observations": int(
                    values.get("securities_with_observations", 0)
                ),
                "securities_with_valid_adj_close": int(
                    values.get("securities_with_valid_adj_close", 0)
                ),
                "securities_with_valid_volume": int(
                    values.get("securities_with_valid_volume", 0)
                ),
                "observations": int(observations),
                "valid_adj_close_sessions": int(
                    values.get("valid_adj_close_sessions", 0)
                ),
                "missing_adj_close_observations": int(
                    values.get("missing_adj_close_observations", 0)
                ),
                "missing_volume_observations": int(
                    values.get("missing_volume_observations", 0)
                ),
                "extreme_return_rows": int(values.get("extreme_return_rows", 0)),
                "raw_adjusted_return_conflict_rows": int(
                    values.get("raw_adjusted_return_conflict_rows", 0)
                ),
                "extreme_volume_spike_rows": int(
                    values.get("extreme_volume_spike_rows", 0)
                ),
                "split_events": int(values.get("split_events", 0)),
                "dividend_events": int(values.get("dividend_events", 0)),
                "expected_sessions_in_observed_intervals": int(expected),
                "unobserved_sessions_within_security_intervals": int(missing),
                "row_missing_adj_close_rate": (
                    values.get("missing_adj_close_observations", 0) / observations
                    if observations
                    else np.nan
                ),
                "row_missing_volume_rate": (
                    values.get("missing_volume_observations", 0) / observations
                    if observations
                    else np.nan
                ),
                "calendar_missing_rate_observed_intervals": (
                    missing / expected if expected else np.nan
                ),
            }
        )
    year_quality = pd.DataFrame(year_rows, columns=YEAR_QUALITY_COLUMNS)
    issues = runtime.issues.frame()

    atomic_write_csv(security_quality, runtime.output_dir / "data_quality_by_security.csv")
    atomic_write_csv(year_quality, runtime.output_dir / "data_quality_by_year.csv")
    atomic_write_csv(issues, runtime.output_dir / "data_issues.csv")
    atomic_write_csv(
        identity_findings,
        runtime.output_dir / "security_identity_findings.csv",
    )
    all_days = pd.date_range(start, end, freq="D")
    weekdays = all_days[all_days.weekday < 5]
    calendar_summary = pd.DataFrame(
        [
            {
                "calendar": config["validation"]["us_calendar"],
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "all_calendar_days": len(all_days),
                "expected_sessions": len(calendar_sessions),
                "weekend_days": int((all_days.weekday >= 5).sum()),
                "weekday_market_holidays": int(
                    len(weekdays.difference(calendar_sessions))
                ),
            }
        ]
    )
    atomic_write_csv(calendar_summary, runtime.output_dir / "calendar_summary.csv")
    atomic_write_json(
        {
            "completed_at_utc": utc_now(),
            "data_mode": data_mode,
            "manifest_source": (
                str(project_path(manifest_path))
                if manifest_path
                else str(metadata_path / "download_manifest.csv")
            ),
            "failures_source": (
                str(project_path(failures_path))
                if failures_path
                else str(metadata_path / "download_failures.csv")
            ),
            "raw_files_processed": len(processed_paths),
            "base_rows": base_sink.rows,
            "diagnostic_rows": diagnostic_sink.rows,
            "standardized_raw_rows": audit_sink.rows,
            "universe_rows_available": len(master),
            "calendar_sessions_expected": len(calendar_sessions),
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "constraints": {
                "final_pct_return_calculated": False,
                "raw_data_overwritten": False,
                "outliers_automatically_deleted": False,
                "missing_values_zero_filled": False,
                "strategy_features_constructed": False,
            },
            "package_versions": {
                name: _package_version(name)
                for name in ("pandas", "numpy", "pyarrow", "exchange-calendars")
            },
            "python": sys.version,
            "platform": platform.platform(),
        },
        runtime.output_dir / "validation_provenance.json",
    )
    return {
        "raw_files_processed": len(processed_paths),
        "base_rows": base_sink.rows,
        "securities_with_raw_data": len(discovered_ids),
        "securities_in_quality_table": len(security_quality),
        "issues": len(issues),
        "output_dir": str(runtime.output_dir),
        "data_mode": data_mode,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--raw-dir", help="Override the configured raw directory.")
    parser.add_argument("--output-dir", help="Override the configured validated directory.")
    parser.add_argument(
        "--master",
        default="data/universe/security_master.parquet",
        help="Security master CSV or Parquet.",
    )
    parser.add_argument(
        "--metadata-dir",
        help="Directory containing download_manifest.csv/download_failures.csv.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional explicit acquisition manifest path.",
    )
    parser.add_argument(
        "--failures",
        help="Optional explicit acquisition failures path.",
    )
    parser.add_argument(
        "--data-mode",
        default="live_or_cli_input",
        help="Provenance label; use 'fixture' for synthetic test data.",
    )
    parser.add_argument("--log-file", help="Optional log file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    configure_logging(args.log_file or project_path(config["paths"]["logs_dir"]) / "validate_data.log")
    result = validate(
        config,
        raw_dir=args.raw_dir or config["paths"]["raw_dir"],
        output_dir=args.output_dir or config["paths"]["validated_dir"],
        master_path=args.master,
        metadata_dir=args.metadata_dir,
        data_mode=args.data_mode,
        manifest_path=args.manifest,
        failures_path=args.failures,
    )
    LOGGER.info("Validation complete: %s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
