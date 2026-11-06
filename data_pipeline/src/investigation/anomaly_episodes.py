"""Group filtered anomaly flags into calendar-aware, reviewable episodes.

The module is read-only with respect to the filtered research panel, validated
raw observations, diagnostic returns, and vendor checkpoints. Temporal grouping
uses official XNYS sessions. Missing sessions are never converted to zero, and
context observations are exported separately from flagged observations.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"),
)

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from src.common import (
    PROJECT_ROOT,
    atomic_write_json,
    configure_logging,
    ensure_parent,
    load_config,
    project_path,
)
from src.filtering.liquidity_screen import daily_dollar_volume


LOGGER = configure_logging(PROJECT_ROOT / "logs" / "filtered_anomaly_episodes.log")

KEY_COLUMNS = ["security_id", "date"]
ANOMALY_FIELDS: dict[str, str] = {
    "zero_volume": "zero_volume_flag",
    "extreme_volume": "extreme_volume_flag",
    "extreme_price": "extreme_price_flag",
    "extreme_return": "extreme_return_flag",
    "ohlc_consistency": "ohlc_consistency_flag",
    "raw_adjusted_conflict": "raw_adjusted_conflict_flag",
    "negative_volume": "negative_volume_flag",
    "non_session_date": "non_session_date_flag",
    "quarantine_candidate": "quarantine_candidate_flag",
    "corporate_action_discrepancy": "corporate_action_discrepancy_flag",
}
ANOMALY_FIELD_ORDER = list(ANOMALY_FIELDS.values())

PERIOD_COLUMNS = [
    "period_id",
    "security_id",
    "ticker",
    "company_name",
    "exchange",
    "anomaly_types",
    "start_date",
    "end_date",
    "n_trading_sessions",
    "n_flagged_sessions",
    "n_zero_volume_sessions",
    "n_extreme_volume_sessions",
    "n_extreme_price_sessions",
    "n_extreme_return_sessions",
    "min_volume",
    "max_volume",
    "mean_volume",
    "median_volume",
    "min_close",
    "max_close",
    "min_adj_close",
    "max_adj_close",
    "min_diagnostic_return",
    "max_diagnostic_return",
    "cumulative_diagnostic_return",
    "total_reported_share_volume",
    "max_daily_dollar_volume",
    "stock_splits_in_period",
    "dividends_in_period",
    "existing_issue_status",
    "existing_evidence_classification",
]

DAILY_COLUMNS = [
    "period_id",
    "security_id",
    "ticker",
    "company_name",
    "date",
    "is_flagged_session",
    "is_context_session",
    "volume",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "diagnostic_pct_return",
    "daily_dollar_volume",
    "zero_volume_flag",
    "extreme_volume_flag",
    "extreme_price_flag",
    "extreme_return_flag",
    "stock_splits",
    "dividends",
    "vendor_repair_flag",
    "existing_quality_flags",
    "future_split_factor",
    "restored_close",
    "quarantine_candidate_flag",
    "corporate_action_discrepancy_flag",
    "source_row_present",
    "diagnostic_row_present",
    "context_missing_session",
    "session_data_status",
    "anomaly_types_present",
]

REVIEW_COLUMNS = [
    "period_id",
    "ticker",
    "company_name",
    "start_date",
    "end_date",
    "anomaly_types",
    "review_priority",
    "technical_notes",
    "possible_economic_explanation",
    "external_evidence_source",
    "reviewer_conclusion",
    "review_status",
    "recommended_treatment",
]

DIAGNOSTIC_FLAG_COLUMNS = [
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path, pattern: str = "*.parquet") -> str:
    """Return a deterministic digest for raw checkpoint files."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".csv",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def future_split_factor(stock_splits: pd.Series) -> pd.Series:
    """Return the product of positive split factors strictly after each row."""
    factors = pd.to_numeric(stock_splits, errors="coerce").copy()
    factors = factors.where(factors.gt(0), 1.0).astype("float64")
    inclusive_reverse_product = factors.iloc[::-1].cumprod().iloc[::-1]
    return inclusive_reverse_product.div(factors)


def restore_compatible_close(
    frame: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Restore contemporaneous Close while preserving reported share volume."""
    required = {"security_id", "date", "close", "volume", "stock_splits"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Raw frame is missing {missing}")
    ordered = frame.sort_values(["security_id", "date"], kind="stable").copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    end = pd.Timestamp(end_date).normalize()
    factors = []
    for _, group in ordered.groupby("security_id", sort=False):
        split_factors = group["stock_splits"].where(
            group["date"].le(end), 1.0
        )
        factors.append(future_split_factor(split_factors))
    ordered["future_split_factor"] = pd.concat(factors).sort_index()
    close = pd.to_numeric(ordered["close"], errors="coerce")
    ordered["restored_close"] = np.where(
        np.isfinite(close) & close.gt(0),
        close * ordered["future_split_factor"],
        np.nan,
    )
    ordered["daily_dollar_volume"] = daily_dollar_volume(
        ordered["restored_close"],
        pd.to_numeric(ordered["volume"], errors="coerce"),
    )
    return ordered.sort_values(["security_id", "date"], kind="stable").reset_index(
        drop=True
    )


def stable_period_id(security_id: str, start_date: pd.Timestamp) -> str:
    """Create a deterministic identifier from security and episode start."""
    start = pd.Timestamp(start_date).normalize()
    token = hashlib.sha256(
        f"{security_id}|{start.date().isoformat()}".encode("utf-8")
    ).hexdigest()[:10].upper()
    return f"AE-{start:%Y%m%d}-{token}"


def assign_episode_keys(
    flagged: pd.DataFrame,
    calendar_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assign one strict episode to each contiguous run of flagged sessions.

    A run continues only when the current and previous flagged rows are adjacent
    official exchange sessions for the same security. Weekends and holidays are
    skipped; an intervening unflagged or absent exchange session breaks a run.
    """
    columns = [
        *KEY_COLUMNS,
        "calendar_position",
        "episode_number",
        "start_date",
        "period_id",
    ]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    if flagged.duplicated(KEY_COLUMNS).any():
        raise ValueError("Flagged rows contain duplicate security-date keys")
    ordered = flagged[KEY_COLUMNS].copy()
    ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
    sessions = pd.DatetimeIndex(calendar_sessions).normalize()
    positions = pd.Series(np.arange(len(sessions), dtype=int), index=sessions)
    ordered["calendar_position"] = ordered["date"].map(positions)
    ordered = ordered.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    same_security = ordered["security_id"].eq(ordered["security_id"].shift())
    adjacent = ordered["calendar_position"].sub(
        ordered["calendar_position"].shift()
    ).eq(1)
    starts = ~(same_security & adjacent)
    ordered["episode_number"] = starts.cumsum()
    starts_frame = ordered.groupby("episode_number", sort=False)["date"].min()
    ordered["start_date"] = ordered["episode_number"].map(starts_frame)
    ordered["period_id"] = [
        stable_period_id(security_id, start)
        for security_id, start in zip(
            ordered["security_id"], ordered["start_date"], strict=True
        )
    ]
    episode_headers = ordered.drop_duplicates("episode_number", keep="first")
    if episode_headers["period_id"].duplicated().any():
        duplicates = episode_headers.loc[
            episode_headers["period_id"].duplicated(keep=False),
            ["security_id", "date", "start_date", "period_id"],
        ]
        raise ValueError(
            "Generated period_id values are not unique: "
            f"{duplicates.head(10).to_dict('records')}"
        )
    return ordered


def _session_keys(
    calendar: Any,
    security_id: Any,
    start: Any,
    end: Any,
) -> list[tuple[str, pd.Timestamp]]:
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    return [
        (str(security_id), session.normalize())
        for session in calendar.sessions_in_range(first, last)
    ]


def _price_level_evidence(
    price_cases: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], str]:
    evidence: dict[tuple[str, pd.Timestamp], str] = {}
    if price_cases.empty:
        return evidence
    for row in price_cases.itertuples(index=False):
        classification = str(getattr(row, "classification", "") or "")
        if not classification or classification == "not_classified_extreme":
            continue
        for field in ("min_date", "max_date"):
            value = getattr(row, field, pd.NaT)
            if pd.notna(value):
                evidence[(str(row.security_id), pd.Timestamp(value).normalize())] = (
                    classification
                )
    return evidence


def _corporate_action_evidence(
    split_reconciliation: pd.DataFrame,
    dividend_reconciliation: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], str]:
    evidence: dict[tuple[str, pd.Timestamp], str] = {}
    frames = [
        split_reconciliation.assign(event_kind="split_reconciliation"),
        dividend_reconciliation.assign(event_kind="dividend_reconciliation"),
    ]
    for frame in frames:
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            key = (str(row.security_id), pd.Timestamp(row.event_date).normalize())
            text = (
                f"{row.event_kind}:{row.final_classification}:"
                f"{row.resolution_status}"
            )
            if key in evidence:
                evidence[key] = f"{evidence[key]}|{text}"
            else:
                evidence[key] = text
    return evidence


def _quarantine_evidence(
    quarantine: pd.DataFrame,
    calendar: Any,
) -> tuple[set[tuple[str, pd.Timestamp]], dict[tuple[str, pd.Timestamp], str]]:
    keys: set[tuple[str, pd.Timestamp]] = set()
    evidence: dict[tuple[str, pd.Timestamp], str] = {}
    for row in quarantine.itertuples(index=False):
        for key in _session_keys(
            calendar, row.security_id, row.start_date, row.end_date
        ):
            keys.add(key)
            text = f"quarantine:{row.scope}:{row.reason}"
            evidence[key] = (
                f"{evidence[key]}|{text}" if key in evidence else text
            )
    return keys, evidence


def build_anomaly_panel(
    raw: pd.DataFrame,
    diagnostics: pd.DataFrame,
    master: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
    calendar: Any,
    price_cases: pd.DataFrame,
    quarantine: pd.DataFrame,
    split_reconciliation: pd.DataFrame,
    dividend_reconciliation: pd.DataFrame,
    extreme_return_cases: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one row per filtered raw stock-day with all anomaly flags."""
    raw_ordered = restore_compatible_close(raw, end_date=end_date)
    diagnostic_columns = [
        "security_id",
        "date",
        *DIAGNOSTIC_FLAG_COLUMNS,
        "diagnostic_pct_return",
    ]
    missing = sorted(set(diagnostic_columns) - set(diagnostics.columns))
    if missing:
        raise ValueError(f"Diagnostic table is missing {missing}")
    diagnostic_frame = diagnostics[diagnostic_columns].copy()
    diagnostic_frame["date"] = pd.to_datetime(diagnostic_frame["date"]).dt.normalize()
    panel = raw_ordered.merge(
        diagnostic_frame,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator="_diagnostic_merge",
    )
    panel["diagnostic_row_present"] = panel["_diagnostic_merge"].eq("both")
    panel = panel.drop(columns="_diagnostic_merge")

    for column in DIAGNOSTIC_FLAG_COLUMNS:
        panel[column] = _as_bool(panel[column])

    price_evidence = _price_level_evidence(price_cases)
    keys = list(zip(panel["security_id"].astype(str), panel["date"], strict=True))
    panel["extreme_price_evidence"] = [price_evidence.get(key, "") for key in keys]
    panel["extreme_price_flag"] = (
        panel["zero_or_negative_adj_close_flag"].astype(bool)
        | panel["extreme_price_evidence"].ne("")
    )

    corp_evidence = _corporate_action_evidence(
        split_reconciliation, dividend_reconciliation
    )
    panel["corporate_action_evidence"] = [
        corp_evidence.get(key, "") for key in keys
    ]
    panel["corporate_action_discrepancy_flag"] = panel[
        "corporate_action_evidence"
    ].ne("")

    quarantine_keys, quarantine_evidence = _quarantine_evidence(
        quarantine, calendar
    )
    panel["quarantine_candidate_flag"] = [key in quarantine_keys for key in keys]
    panel["quarantine_evidence"] = [
        quarantine_evidence.get(key, "") for key in keys
    ]

    return_evidence: dict[tuple[str, pd.Timestamp], str] = {}
    vendor_repairs: dict[tuple[str, pd.Timestamp], bool] = {}
    if not extreme_return_cases.empty:
        for row in extreme_return_cases.itertuples(index=False):
            key = (str(row.security_id), pd.Timestamp(row.date).normalize())
            return_evidence[key] = str(row.classification)
            vendor_repairs[key] = bool(row.vendor_repaired)
    panel["extreme_return_classification"] = [
        return_evidence.get(key, "") for key in keys
    ]
    panel["vendor_repair_flag"] = [vendor_repairs.get(key, False) for key in keys]

    panel["extreme_volume_flag"] = panel["extreme_volume_spike_flag"]
    panel["raw_adjusted_conflict_flag"] = panel[
        "raw_adjusted_return_conflict_flag"
    ]
    panel["ohlc_consistency_flag"] = panel["ohlc_consistency_flag"]
    panel["negative_volume_flag"] = panel["negative_volume_flag"]
    panel["non_session_date_flag"] = panel["non_session_date_flag"]

    panel["any_anomaly_flag"] = panel[ANOMALY_FIELD_ORDER].any(axis=1)
    panel["existing_quality_flags"] = panel.apply(
        _existing_quality_flag_text, axis=1
    )

    metadata = master[
        ["security_id", "ticker_2023", "company_name", "exchange"]
    ].rename(columns={"ticker_2023": "master_ticker"})
    panel = panel.merge(
        metadata, on="security_id", how="left", validate="many_to_one"
    )
    panel["ticker"] = panel["master_ticker"].fillna(panel["ticker"])
    return panel.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def _existing_quality_flag_text(row: pd.Series) -> str:
    labels = []
    flag_labels = {
        "zero_volume_flag": "zero_volume",
        "extreme_volume_spike_flag": "extreme_volume",
        "price_discontinuity_flag": "price_discontinuity",
        "raw_adjusted_return_conflict_flag": "raw_adjusted_conflict",
        "ohlc_consistency_flag": "ohlc_consistency",
        "zero_or_negative_adj_close_flag": "zero_or_negative_adj_close",
        "negative_volume_flag": "negative_volume",
        "non_session_date_flag": "non_session_date",
        "extreme_price_flag": "extreme_price_evidence",
        "quarantine_candidate_flag": "quarantine_candidate",
        "corporate_action_discrepancy_flag": "corporate_action_discrepancy",
    }
    for column, label in flag_labels.items():
        if bool(row.get(column, False)):
            labels.append(label)
    return ";".join(labels)


def _nonempty_unique(values: Iterable[Any]) -> str:
    items: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        for part in str(value).split("|"):
            part = part.strip()
            if part:
                items.add(part)
    return ";".join(sorted(items))


def _overlapping_issue_status(
    issues: pd.DataFrame,
    *,
    security_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    if issues.empty:
        return ""
    relevant = issues[issues["security_id"].astype(str).eq(str(security_id))].copy()
    if relevant.empty:
        return ""
    first = pd.to_datetime(relevant["first_date"], errors="coerce")
    last = pd.to_datetime(relevant["last_date"], errors="coerce")
    overlap = first.le(end) & last.ge(start)
    values = [
        f"{row.issue_id}:{row.issue_type}:{row.status}"
        for row in relevant.loc[overlap].itertuples(index=False)
    ]
    return ";".join(sorted(set(values)))


def build_periods(
    panel: pd.DataFrame,
    calendar_sessions: pd.DatetimeIndex,
    issues: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one summary row for every strict anomaly episode."""
    flagged = panel[panel["any_anomaly_flag"]].copy()
    keys = assign_episode_keys(flagged, calendar_sessions)
    flagged = flagged.merge(keys, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if flagged["period_id"].isna().any():
        raise ValueError("At least one flagged observation has no episode ID")

    records: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    sessions = pd.DatetimeIndex(calendar_sessions).normalize()
    for period_id, group in flagged.groupby("period_id", sort=False):
        group = group.sort_values("date", kind="stable")
        first = group["date"].min()
        last = group["date"].max()
        type_counts = {
            label: int(group[field].astype(bool).sum())
            for label, field in ANOMALY_FIELDS.items()
        }
        anomaly_types = [
            label for label, count in type_counts.items() if count > 0
        ]
        returns = pd.to_numeric(
            group["diagnostic_pct_return"], errors="coerce"
        ).dropna()
        cumulative = np.nan
        if (
            len(returns) == len(group)
            and np.isfinite(returns).all()
            and returns.gt(-1).all()
        ):
            cumulative = float(np.prod(1.0 + returns.to_numpy()) - 1.0)
        first_position = int(sessions.get_loc(first))
        last_position = int(sessions.get_loc(last))
        n_sessions = last_position - first_position + 1
        if n_sessions != len(group):
            raise ValueError(f"Episode {period_id} is not strictly contiguous")
        volume = pd.to_numeric(group["volume"], errors="coerce")
        close = pd.to_numeric(group["close"], errors="coerce")
        adjusted = pd.to_numeric(group["adj_close"], errors="coerce")
        diagnostic = pd.to_numeric(
            group["diagnostic_pct_return"], errors="coerce"
        )
        issue_status = _overlapping_issue_status(
            issues,
            security_id=str(group.iloc[0]["security_id"]),
            start=first,
            end=last,
        )
        evidence = _nonempty_unique(
            pd.concat(
                [
                    group["extreme_return_classification"],
                    group["extreme_price_evidence"],
                    group["quarantine_evidence"],
                    group["corporate_action_evidence"],
                ],
                ignore_index=True,
            )
        )
        stock_splits = int(
            pd.to_numeric(group["stock_splits"], errors="coerce").gt(0).sum()
        )
        dividends = int(
            pd.to_numeric(group["dividends"], errors="coerce").gt(0).sum()
        )
        unresolved_tokens = (
            "unresolved",
            "near_zero_denominator",
            "vendor_repaired",
            "raw_adjusted",
            "ohlc_consistency",
            "zero_or_negative_adj_close",
        )
        has_unresolved_quality = any(
            token in evidence.lower() for token in unresolved_tokens
        ) or bool(group["quarantine_candidate_flag"].any())
        if any(
            prefix in evidence.lower()
            for prefix in ("split_reconciliation:", "dividend_reconciliation:")
        ):
            has_unresolved_quality = has_unresolved_quality or any(
                token in evidence.lower()
                for token in ("partially_resolved", "adjustment_mismatch")
            )
        has_unresolved_quality = has_unresolved_quality or any(
            token in issue_status.lower()
            for token in (
                "ohlc_consistency:open",
                "raw_adjusted_return_conflict:open",
                "zero_or_negative_adj_close:open",
                "corporate_action:open",
                "corporate_action:unresolved",
            )
        )
        records.append(
            {
                "period_id": period_id,
                "security_id": str(group.iloc[0]["security_id"]),
                "ticker": group.iloc[0]["ticker"],
                "company_name": group.iloc[0]["company_name"],
                "exchange": group.iloc[0]["exchange"],
                "anomaly_types": ";".join(anomaly_types),
                "start_date": first.date().isoformat(),
                "end_date": last.date().isoformat(),
                "n_trading_sessions": n_sessions,
                "n_flagged_sessions": len(group),
                "n_zero_volume_sessions": type_counts["zero_volume"],
                "n_extreme_volume_sessions": type_counts["extreme_volume"],
                "n_extreme_price_sessions": type_counts["extreme_price"],
                "n_extreme_return_sessions": type_counts["extreme_return"],
                "min_volume": volume.min(),
                "max_volume": volume.max(),
                "mean_volume": volume.mean(),
                "median_volume": volume.median(),
                "min_close": close.min(),
                "max_close": close.max(),
                "min_adj_close": adjusted.min(),
                "max_adj_close": adjusted.max(),
                "min_diagnostic_return": diagnostic.min(),
                "max_diagnostic_return": diagnostic.max(),
                "cumulative_diagnostic_return": cumulative,
                "total_reported_share_volume": (
                    float(volume.sum()) if volume.notna().any() else np.nan
                ),
                "max_daily_dollar_volume": pd.to_numeric(
                    group["daily_dollar_volume"], errors="coerce"
                ).max(),
                "stock_splits_in_period": stock_splits,
                "dividends_in_period": dividends,
                "existing_issue_status": issue_status,
                "existing_evidence_classification": evidence,
            }
        )
        internal.append(
            {
                "period_id": period_id,
                "touches_liquidity_window": bool(
                    group["date"].isin(
                        sessions[
                            (sessions >= pd.Timestamp("2022-12-02"))
                            & (sessions <= pd.Timestamp("2022-12-30"))
                        ]
                    ).any()
                ),
                "has_multiple_anomaly_types": len(anomaly_types) > 1,
                "has_unresolved_quality": has_unresolved_quality,
                "has_corporate_action": stock_splits > 0
                or dividends > 0
                or "corporate_action_discrepancy" in anomaly_types,
                "all_extreme_returns_retained_market_moves": bool(
                    type_counts["extreme_return"]
                    and set(
                        group.loc[
                            group["extreme_return_flag"],
                            "extreme_return_classification",
                        ]
                        .dropna()
                        .astype(str)
                        .loc[lambda values: values.ne("")]
                    )
                    == {"retained_extreme_market_move"}
                ),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(internal), flagged


def assign_review_priority(
    period: pd.Series,
    internal: pd.Series,
    *,
    material_daily_dollar_volume_usd: float,
    multi_session_zero_volume_min_sessions: int,
) -> str:
    """Assign a transparent review priority without excluding observations."""
    if bool(internal["touches_liquidity_window"]):
        return "P0_liquidity_window"
    if bool(internal["has_unresolved_quality"]):
        return "P1_unresolved_quality"
    if bool(internal["has_multiple_anomaly_types"]):
        return "P1_overlapping_anomaly_types"
    if (
        int(period["n_zero_volume_sessions"])
        >= multi_session_zero_volume_min_sessions
    ):
        return "P1_multi_session_zero_volume"
    if (
        pd.notna(period["max_daily_dollar_volume"])
        and float(period["max_daily_dollar_volume"])
        >= material_daily_dollar_volume_usd
        and int(period["n_extreme_volume_sessions"]) > 0
    ):
        return "P1_material_extreme_volume"
    if bool(internal["has_corporate_action"]):
        return "P2_corporate_action_review"
    if bool(internal["all_extreme_returns_retained_market_moves"]):
        return "P3_isolated_retained_market_move"
    return "P2_anomaly_review"


def build_review_register(
    periods: pd.DataFrame,
    internal: pd.DataFrame,
    *,
    material_daily_dollar_volume_usd: float,
    multi_session_zero_volume_min_sessions: int,
) -> pd.DataFrame:
    """Create the human-review queue, leaving interpretation fields blank."""
    merged = periods.merge(internal, on="period_id", validate="one_to_one")
    rows = []
    for row in merged.itertuples(index=False):
        period = pd.Series(row._asdict())
        internal_row = period[[
            "touches_liquidity_window",
            "has_multiple_anomaly_types",
            "has_unresolved_quality",
            "has_corporate_action",
            "all_extreme_returns_retained_market_moves",
        ]]
        priority = assign_review_priority(
            period,
            internal_row,
            material_daily_dollar_volume_usd=material_daily_dollar_volume_usd,
            multi_session_zero_volume_min_sessions=(
                multi_session_zero_volume_min_sessions
            ),
        )
        notes = (
            f"Strict XNYS episode from {row.start_date} through {row.end_date}: "
            f"{row.n_flagged_sessions} flagged sessions; types="
            f"{row.anomaly_types}. Flag counts: zero-volume="
            f"{row.n_zero_volume_sessions}, extreme-volume="
            f"{row.n_extreme_volume_sessions}, extreme-price="
            f"{row.n_extreme_price_sessions}, extreme-return="
            f"{row.n_extreme_return_sessions}. Diagnostic return range="
            f"{row.min_diagnostic_return!r} to {row.max_diagnostic_return!r}; "
            f"cumulative="
            f"{row.cumulative_diagnostic_return!r}"
            f"{' (null: at least one flagged session lacks a valid diagnostic return)' if pd.isna(row.cumulative_diagnostic_return) else ''}. "
            f"Corporate actions on flagged sessions: splits="
            f"{row.stock_splits_in_period}, dividends="
            f"{row.dividends_in_period}. Existing evidence/status="
            f"{row.existing_evidence_classification or 'none'}; "
            f"{row.existing_issue_status or 'no overlapping issue record'}."
        )
        rows.append(
            {
                "period_id": row.period_id,
                "ticker": row.ticker,
                "company_name": row.company_name,
                "start_date": row.start_date,
                "end_date": row.end_date,
                "anomaly_types": row.anomaly_types,
                "review_priority": priority,
                "technical_notes": notes,
                "possible_economic_explanation": "",
                "external_evidence_source": "",
                "reviewer_conclusion": "",
                "review_status": "pending_review",
                "recommended_treatment": "",
            }
        )
    review = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    review["_rank"] = review["review_priority"].str[:2].map(rank).fillna(9)
    review = review.sort_values(
        ["_rank", "start_date", "ticker", "period_id"], kind="stable"
    ).drop(columns="_rank")
    return review.reset_index(drop=True)


def build_daily_context(
    periods: pd.DataFrame,
    panel: pd.DataFrame,
    flagged: pd.DataFrame,
    calendar_sessions: pd.DatetimeIndex,
    *,
    before: int,
    after: int,
) -> pd.DataFrame:
    """Export every XNYS context session for each episode."""
    sessions = pd.DatetimeIndex(calendar_sessions).normalize()
    flagged_by_period = {
        period_id: group.sort_values("date")
        for period_id, group in flagged.groupby("period_id", sort=False)
    }
    panel_by_security = {
        security_id: group.set_index("date", drop=False).sort_index()
        for security_id, group in panel.groupby("security_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for episode in periods.itertuples(index=False):
        start = pd.Timestamp(episode.start_date)
        end = pd.Timestamp(episode.end_date)
        start_position = int(sessions.get_loc(start))
        end_position = int(sessions.get_loc(end))
        context_start = max(0, start_position - before)
        context_end = min(len(sessions) - 1, end_position + after)
        context_sessions = sessions[context_start : context_end + 1]
        flagged_dates = set(flagged_by_period[episode.period_id]["date"])
        panel = panel_by_security.get(str(episode.security_id), pd.DataFrame())
        for session in context_sessions:
            is_flagged = session in flagged_dates
            if session in panel.index:
                source = panel.loc[session]
                if isinstance(source, pd.DataFrame):
                    source = source.iloc[0]
                source_present = True
            else:
                source = None
                source_present = False
            values = {
                "volume": np.nan,
                "open": np.nan,
                "high": np.nan,
                "low": np.nan,
                "close": np.nan,
                "adj_close": np.nan,
                "diagnostic_pct_return": np.nan,
                "daily_dollar_volume": np.nan,
                "zero_volume_flag": False,
                "extreme_volume_flag": False,
                "extreme_price_flag": False,
                "extreme_return_flag": False,
                "stock_splits": np.nan,
                "dividends": np.nan,
                "vendor_repair_flag": False,
                "future_split_factor": np.nan,
                "restored_close": np.nan,
                "quarantine_candidate_flag": False,
                "corporate_action_discrepancy_flag": False,
                "diagnostic_row_present": False,
                "existing_quality_flags": "",
                "anomaly_types_present": "",
            }
            if source is not None:
                for column in values:
                    if column in source.index:
                        values[column] = source[column]
                values["zero_volume_flag"] = bool(source["zero_volume_flag"])
                values["extreme_volume_flag"] = bool(source["extreme_volume_flag"])
                values["extreme_price_flag"] = bool(source["extreme_price_flag"])
                values["extreme_return_flag"] = bool(source["extreme_return_flag"])
                values["vendor_repair_flag"] = bool(source["vendor_repair_flag"])
                values["quarantine_candidate_flag"] = bool(
                    source["quarantine_candidate_flag"]
                )
                values["corporate_action_discrepancy_flag"] = bool(
                    source["corporate_action_discrepancy_flag"]
                )
                values["diagnostic_row_present"] = bool(
                    source["diagnostic_row_present"]
                )
                present_types = [
                    label
                    for label, field in ANOMALY_FIELDS.items()
                    if bool(source[field])
                ]
                values["anomaly_types_present"] = ";".join(present_types)
            all_null_prices = source_present and all(
                pd.isna(values[column])
                for column in ("open", "high", "low", "close", "adj_close")
            )
            if is_flagged:
                status = "flagged_source_row"
            elif not source_present:
                status = "missing_context_session"
            elif all_null_prices:
                status = "placeholder_context_row"
            else:
                status = "available_context_session"
            rows.append(
                {
                    "period_id": episode.period_id,
                    "security_id": episode.security_id,
                    "ticker": episode.ticker,
                    "company_name": episode.company_name,
                    "date": session.date().isoformat(),
                    "is_flagged_session": is_flagged,
                    "is_context_session": not is_flagged,
                    **values,
                    "source_row_present": source_present,
                    "context_missing_session": bool(
                        not is_flagged and not source_present
                    ),
                    "session_data_status": status,
                }
            )
    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


def build_episode_summary(
    periods: pd.DataFrame,
    internal: pd.DataFrame,
    review: pd.DataFrame,
    flagged: pd.DataFrame,
) -> pd.DataFrame:
    """Create the aggregate episode summary as a metric-value table."""
    metrics = periods.merge(internal, on="period_id", validate="one_to_one")
    rows: list[dict[str, Any]] = []

    def add(
        metric: str,
        value: int | float,
        *,
        category: str = "",
        affected_securities: int | None = None,
        flagged_observations: int | None = None,
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "metric": metric,
                "category": category,
                "value": int(value) if float(value).is_integer() else float(value),
                "affected_securities": affected_securities,
                "flagged_observations": flagged_observations,
                "notes": notes,
            }
        )

    add(
        "anomaly_episodes",
        len(periods),
        affected_securities=int(periods["security_id"].nunique()),
        flagged_observations=len(flagged),
        notes="Strict XNYS-session episodes; multi-label dates counted once.",
    )
    for label, field in ANOMALY_FIELDS.items():
        mask = periods["anomaly_types"].str.split(";").map(
            lambda values, target=label: target in values
        )
        add(
            "episodes_by_anomaly_type",
            int(mask.sum()),
            category=label,
            affected_securities=int(periods.loc[mask, "security_id"].nunique()),
            flagged_observations=int(flagged[field].astype(bool).sum()),
            notes=(
                "Category counts overlap when an episode has multiple anomaly "
                "types."
            ),
        )

    duration_bins = [
        ("one_session", 1, 1),
        ("two_to_five_sessions", 2, 5),
        ("six_to_twenty_sessions", 6, 20),
        ("more_than_twenty_sessions", 21, int(periods["n_trading_sessions"].max()) if len(periods) else 21),
    ]
    for label, low, high in duration_bins:
        mask = periods["n_trading_sessions"].between(low, high, inclusive="both")
        add(
            "episode_duration",
            int(mask.sum()),
            category=label,
            affected_securities=int(periods.loc[mask, "security_id"].nunique()),
            flagged_observations=int(
                periods.loc[mask, "n_flagged_sessions"].sum()
            ),
        )
    add(
        "episodes_with_multiple_anomaly_types",
        int(metrics["has_multiple_anomaly_types"].sum()),
        affected_securities=int(
            metrics.loc[metrics["has_multiple_anomaly_types"], "security_id"].nunique()
        ),
    )
    add(
        "episodes_involving_corporate_actions",
        int(metrics["has_corporate_action"].sum()),
        affected_securities=int(
            metrics.loc[metrics["has_corporate_action"], "security_id"].nunique()
        ),
    )
    add(
        "episodes_with_unresolved_quality_issues",
        int(metrics["has_unresolved_quality"].sum()),
        affected_securities=int(
            metrics.loc[metrics["has_unresolved_quality"], "security_id"].nunique()
        ),
    )
    add(
        "episodes_affecting_liquidity_screen_window",
        int(metrics["touches_liquidity_window"].sum()),
        affected_securities=int(
            metrics.loc[metrics["touches_liquidity_window"], "security_id"].nunique()
        ),
    )
    high = review["review_priority"].str.startswith(("P0", "P1"))
    add(
        "high_priority_review_episodes",
        int(high.sum()),
        affected_securities=int(review.loc[high, "ticker"].nunique()),
        notes="P0 and P1 review-priority episodes.",
    )
    return pd.DataFrame(rows)


def _build_checks(
    periods: pd.DataFrame,
    flagged: pd.DataFrame,
    daily: pd.DataFrame,
    panel: pd.DataFrame,
    expected: dict[str, int],
    hashes_match: bool,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    flag_keys = set(
        zip(
            flagged["security_id"].astype(str),
            flagged["date"],
            strict=True,
        )
    )
    add(
        "flagged_observations_have_one_episode",
        flagged["period_id"].notna().all()
        and not flagged.duplicated(KEY_COLUMNS).any(),
        f"{len(flag_keys)} flagged security-date keys",
    )
    add(
        "episodes_belong_to_one_security",
        periods["period_id"].is_unique
        and periods["security_id"].notna().all(),
        f"{periods['security_id'].nunique()} securities",
    )
    start_end_valid = all(
        (
            not flagged[
                flagged["period_id"].eq(row.period_id)
                & flagged["date"].eq(pd.Timestamp(row.start_date))
            ].empty
            and not flagged[
                flagged["period_id"].eq(row.period_id)
                & flagged["date"].eq(pd.Timestamp(row.end_date))
            ].empty
        )
        for row in periods.itertuples(index=False)
    )
    add(
        "episode_boundaries_are_flagged_sessions",
        start_end_valid,
        "Every start and end date is present in its episode flag set.",
    )
    add(
        "context_rows_distinguish_flagged_sessions",
        bool(
            (
                daily["is_flagged_session"]
                == daily["session_data_status"].eq("flagged_source_row")
            ).all()
            and (
                daily["is_context_session"]
                == ~daily["is_flagged_session"]
            ).all()
        ),
        f"{len(daily)} context rows",
    )
    zero_expected = expected["zero_volume_flag"]
    zero_actual = int(flagged["zero_volume_flag"].sum())
    add(
        "zero_volume_rows_reconcile",
        zero_actual == zero_expected,
        f"episode={zero_actual}; diagnostic={zero_expected}",
    )
    volume_expected = expected["extreme_volume_spike_flag"]
    volume_actual = int(flagged["extreme_volume_flag"].sum())
    add(
        "extreme_volume_rows_reconcile",
        volume_actual == volume_expected,
        f"episode={volume_actual}; diagnostic={volume_expected}",
    )
    return_expected = expected["extreme_return_flag"]
    return_actual = int(flagged["extreme_return_flag"].sum())
    add(
        "extreme_return_rows_reconcile",
        return_actual == return_expected,
        f"episode={return_actual}; diagnostic={return_expected}",
    )
    diagnostic_mask = panel[DIAGNOSTIC_FLAG_COLUMNS].fillna(False).any(axis=1)
    diagnostic_keys = set(
        zip(
            panel.loc[diagnostic_mask, "security_id"].astype(str),
            panel.loc[diagnostic_mask, "date"],
            strict=True,
        )
    )
    add(
        "all_diagnostic_flag_rows_assigned",
        diagnostic_keys.issubset(flag_keys),
        (
            f"diagnostic={len(diagnostic_keys)}; assigned={len(flag_keys)}; "
            f"unassigned={len(diagnostic_keys - flag_keys)}"
        ),
    )
    missing_sessions_zero = panel.loc[
        ~panel["diagnostic_row_present"], "volume"
    ]
    add(
        "missing_rows_not_converted_to_zero",
        bool(missing_sessions_zero.isna().all()),
        f"{int(missing_sessions_zero.notna().sum())} missing diagnostic rows "
        "carry non-null volume",
    )
    add(
        "protected_inputs_unchanged",
        hashes_match,
        "Selected and validated input hashes match before/after.",
    )
    return pd.DataFrame(checks)


def _expected_flag_counts(diagnostics_selected: pd.DataFrame) -> dict[str, int]:
    counts = {
        column: int(_as_bool(diagnostics_selected[column]).sum())
        for column in DIAGNOSTIC_FLAG_COLUMNS
    }
    counts["any_diagnostic_flag"] = int(
        diagnostics_selected[DIAGNOSTIC_FLAG_COLUMNS]
        .fillna(False)
        .any(axis=1)
        .sum()
    )
    return counts


def _write_readme(
    path: Path,
    *,
    periods: pd.DataFrame,
    summary: pd.DataFrame,
    review: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    high = int(review["review_priority"].str.startswith(("P0", "P1")).sum())
    category_rows = summary[
        summary["metric"].eq("episodes_by_anomaly_type")
    ]
    category_lines = "\n".join(
        f"- {row.category}: {int(row.value):,} episodes, "
        f"{int(row.flagged_observations):,} flagged stock-days"
        for row in category_rows.itertuples(index=False)
    )
    text = f"""# Filtered anomaly episode methodology

This directory contains a diagnostic event-period extraction from the fixed
ADTV-filtered cohort. The analysis does not modify vendor, validated, filtered,
or source issue-register files and initiates no downloads.

Episodes use strict adjacency on official XNYS sessions. Weekends and exchange
holidays are skipped; an intervening unflagged or absent exchange session ends
an episode. Each flagged stock-day belongs to one episode and one security.
Missing observations are preserved as null and are never converted to zero.

Required anomaly categories are zero volume, existing extreme-volume flags,
existing extreme adjusted-price evidence, and existing diagnostic returns
above the absolute 50 percent review threshold. Additional traceable labels
preserve OHLC consistency errors, quarantine windows, and the four selected
corporate-action reconciliation records. Quarantine and corporate-action labels
augment rather than replace the required anomaly categories.

Daily dollar volume restores the contemporaneous price basis by multiplying the
vendor nominal Close by future split factors strictly after each observation and
then multiplying by reported share volume. This follows the liquidity-screen
methodology and does not use future volume or returns.

Cumulative diagnostic return compounds only the flagged sessions in an episode.
It is null if any flagged session lacks a finite diagnostic return or implies a
non-positive return denominator. No return is imputed.

Review priority is technical rather than an exclusion decision. `P0` marks
episodes touching the December 2022 liquidity window. `P1` marks unresolved
price/return or corporate-action evidence, quarantine evidence, overlapping
anomaly types, multi-session zero volume, or an extreme-volume episode whose
maximum daily dollar volume reaches USD 100 million. `P2` covers remaining
corporate-action or anomaly reviews. `P3` covers isolated extreme returns that
the existing evidence classifies as retained market movements.

## Result

- Strict anomaly episodes: {len(periods):,}
- Affected securities: {periods['security_id'].nunique():,}
- High-priority P0/P1 reviews: {high:,}
- Validation checks passed: {int(checks['passed'].sum()):,}/{len(checks):,}
{category_lines}

The context export contains five XNYS sessions before and after each episode,
clipped to 2016-01-04 through 2025-12-31. When context windows overlap, the same
stock-day appears once per referencing `period_id`; this is deliberate and
prevents context from changing the strict episode boundaries.

The review register leaves economic interpretation, external evidence, reviewer
conclusion, and treatment recommendation blank for human completion.
"""
    ensure_parent(path).write_text(text, encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    """Run anomaly episode detection and write all deliverables."""
    config = load_config(project_path(config_path))
    input_cfg = config["inputs"]
    filtered_root = project_path(input_cfg["filtered_root"])
    output_root = project_path(config["outputs"]["root"])
    methodology = project_path(config["outputs"]["methodology"])
    output_root.mkdir(parents=True, exist_ok=True)

    selected_master_path = project_path(input_cfg["selected_master"])
    selected_daily_path = project_path(input_cfg["selected_daily"])
    raw_path = project_path(input_cfg["standardized_raw"])
    diagnostic_path = project_path(input_cfg["diagnostic_returns"])
    protected_files = [
        selected_master_path,
        selected_daily_path,
        raw_path,
        diagnostic_path,
    ]
    hashes_before = {str(path): _sha256(path) for path in protected_files}
    raw_directory = project_path(input_cfg["raw_yahoo_dir"])
    raw_tree_before = tree_sha256(raw_directory) if raw_directory.exists() else None

    master = pd.read_parquet(selected_master_path)
    selected_ids = set(master["security_id"].astype(str))
    selected_daily = pd.read_parquet(
        selected_daily_path, columns=["date", "security_id"]
    )
    LOGGER.info("Loading %d selected securities and %d validated rows",
                len(selected_ids), len(selected_daily))
    diagnostics = pd.read_parquet(
        diagnostic_path,
        columns=["security_id", "date", *DIAGNOSTIC_FLAG_COLUMNS, "diagnostic_pct_return"],
        filters=[("security_id", "in", sorted(selected_ids))],
    )
    raw = pd.read_parquet(
        raw_path,
        columns=[
            "security_id",
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
        ],
        filters=[("security_id", "in", sorted(selected_ids))],
    )
    quality_dir = project_path(input_cfg["filtered_quality_dir"])
    price_cases = pd.read_csv(
        quality_dir / "filtered_extreme_price_cases.csv", low_memory=False
    )
    quarantine = pd.read_csv(
        quality_dir / "filtered_quarantine_candidates.csv", low_memory=False
    )
    split_reconciliation = pd.read_csv(
        quality_dir / "filtered_split_reconciliation.csv", low_memory=False
    )
    dividend_reconciliation = pd.read_csv(
        quality_dir / "filtered_dividend_reconciliation.csv", low_memory=False
    )
    extreme_return_cases = pd.read_csv(
        quality_dir / "filtered_extreme_return_cases.csv", low_memory=False
    )
    issues = pd.read_csv(
        project_path(input_cfg["validation_issues"]), low_memory=False
    )
    issues = issues[issues["security_id"].astype(str).isin(selected_ids)].copy()

    calendar = xcals.get_calendar(config["project"]["calendar"])
    all_sessions = calendar.sessions_in_range(
        config["project"]["start_date"], config["project"]["end_date"]
    )
    panel = build_anomaly_panel(
        raw,
        diagnostics,
        master,
        end_date=pd.Timestamp(config["project"]["end_date"]),
        calendar=calendar,
        price_cases=price_cases,
        quarantine=quarantine,
        split_reconciliation=split_reconciliation,
        dividend_reconciliation=dividend_reconciliation,
        extreme_return_cases=extreme_return_cases,
    )
    periods, internal, flagged = build_periods(panel, all_sessions, issues)
    if periods.empty:
        raise RuntimeError("No anomaly episodes were identified")
    daily = build_daily_context(
        periods,
        panel,
        flagged,
        all_sessions,
        before=int(config["project"]["context_sessions_before"]),
        after=int(config["project"]["context_sessions_after"]),
    )
    review = build_review_register(
        periods,
        internal,
        material_daily_dollar_volume_usd=float(
            config["priority"]["material_daily_dollar_volume_usd"]
        ),
        multi_session_zero_volume_min_sessions=int(
            config["priority"]["multi_session_zero_volume_min_sessions"]
        ),
    )
    summary = build_episode_summary(periods, internal, review, flagged)

    hashes_after = {str(path): _sha256(path) for path in protected_files}
    raw_tree_after = tree_sha256(raw_directory) if raw_directory.exists() else None
    hashes_match = hashes_before == hashes_after and raw_tree_before == raw_tree_after
    expected = _expected_flag_counts(diagnostics)
    checks = _build_checks(
        periods, flagged, daily, panel, expected, hashes_match
    )
    if not bool(checks["passed"].all()):
        failures = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"Anomaly episode validation failed: {failures}")

    _write_csv(periods[PERIOD_COLUMNS], output_root / "anomaly_periods.csv")
    _write_csv(daily, output_root / "anomaly_daily_observations.csv")
    _write_csv(summary, output_root / "anomaly_episode_summary.csv")
    _write_csv(review, output_root / "anomaly_review_register.csv")
    _write_csv(checks, output_root / "anomaly_episode_checks.csv")

    high_priority = int(
        review["review_priority"].str.startswith(("P0", "P1")).sum()
    )
    execution = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": {
            "selected_securities": int(master["security_id"].nunique()),
            "validated_stock_days": int(len(selected_daily)),
            "anomaly_episodes": int(len(periods)),
            "affected_securities": int(periods["security_id"].nunique()),
            "flagged_stock_days": int(len(flagged)),
            "high_priority_episodes": high_priority,
            "context_rows": int(len(daily)),
            "missing_context_sessions": int(daily["context_missing_session"].sum()),
            "overlapping_anomaly_episodes": int(
                internal["has_multiple_anomaly_types"].sum()
            ),
            "corporate_action_episodes": int(
                internal["has_corporate_action"].sum()
            ),
            "unresolved_quality_episodes": int(
                internal["has_unresolved_quality"].sum()
            ),
            "liquidity_window_episodes": int(
                internal["touches_liquidity_window"].sum()
            ),
        },
        "existing_flag_reconciliation": expected,
        "all_checks_passed": bool(checks["passed"].all()),
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": int(len(checks)),
        "protected_input_hashes_sha256": hashes_after,
        "raw_yahoo_tree_sha256": raw_tree_after,
        "raw_yahoo_tree_unchanged": raw_tree_before == raw_tree_after,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "exchange_calendars": getattr(xcals, "__version__", "unknown"),
        },
    }
    atomic_write_json(execution, output_root / "execution_summary.json")
    _write_readme(
        output_root / "README.md",
        periods=periods,
        summary=summary,
        review=review,
        checks=checks,
    )
    methodology.parent.mkdir(parents=True, exist_ok=True)
    methodology.write_text(
        (output_root / "README.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    LOGGER.info(
        "Wrote %d episodes across %d securities (%d high priority)",
        len(periods),
        periods["security_id"].nunique(),
        high_priority,
    )
    return execution


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "anomaly_episodes.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
