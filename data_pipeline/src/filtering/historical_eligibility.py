"""Build the final research panel with historical eligibility corrections.

The source acquisition and validated archives are read-only. FERG and DKNG are
retained as security identities but their research histories begin at explicit,
inclusive eligibility dates. All anomaly diagnostics are recomputed only within
the resulting retained history segments.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from src.investigation.anomaly_episodes import assign_episode_keys


LOGGER = configure_logging(PROJECT_ROOT / "logs" / "historical_eligibility.log")

DAILY_COLUMNS = [
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path, pattern: str = "*.parquet") -> str:
    """Return a deterministic digest for raw checkpoint files."""
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
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


def load_eligibility_policy(path: Path) -> pd.DataFrame:
    """Load and validate the explicit historical eligibility policy."""
    policy = pd.read_csv(path, dtype=str, low_memory=False)
    required = {
        "security_id",
        "ticker",
        "eligible_from",
        "eligible_to",
        "exclusion_reason",
        "policy_type",
        "source_reference",
        "verification_status",
        "decision_date",
    }
    missing = sorted(required - set(policy.columns))
    if missing:
        raise ValueError(f"Eligibility policy is missing columns: {missing}")
    if policy["security_id"].duplicated().any():
        raise ValueError("Eligibility policy contains duplicate security IDs")
    if policy["security_id"].isna().any() or policy["eligible_from"].isna().any():
        raise ValueError("Eligibility policy requires security_id and eligible_from")
    policy["eligible_from"] = pd.to_datetime(
        policy["eligible_from"], errors="raise"
    ).dt.normalize()
    policy["eligible_to"] = pd.to_datetime(
        policy["eligible_to"], errors="coerce"
    ).dt.normalize()
    invalid = policy["eligible_to"].notna() & policy["eligible_to"].lt(
        policy["eligible_from"]
    )
    if invalid.any():
        raise ValueError("Eligibility policy has eligible_to before eligible_from")
    return policy


def apply_eligibility_policy(
    daily: pd.DataFrame,
    policy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply inclusive policy boundaries by security_id, not ticker."""
    required = {"security_id", "date"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Daily frame is missing columns: {missing}")
    selected_ids = set(daily["security_id"].astype(str))
    policy_ids = set(policy["security_id"].astype(str))
    unknown = sorted(policy_ids - selected_ids)
    if unknown:
        raise ValueError(f"Eligibility policy references absent securities: {unknown}")
    if daily.duplicated(["security_id", "date"]).any():
        raise ValueError("Source filtered daily panel has duplicate keys")

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    policy_frame = policy.copy()
    policy_frame["security_id"] = policy_frame["security_id"].astype(str)
    frame["security_id"] = frame["security_id"].astype(str)
    joined = frame.merge(
        policy_frame[
            [
                "security_id",
                "ticker",
                "eligible_from",
                "eligible_to",
                "exclusion_reason",
                "policy_type",
                "source_reference",
                "verification_status",
            ]
        ].rename(columns={"ticker": "policy_ticker"}),
        on="security_id",
        how="left",
        validate="many_to_one",
    )
    eligible_from = joined["eligible_from"].fillna(pd.Timestamp("1900-01-01"))
    eligible_to = joined["eligible_to"].fillna(pd.Timestamp("2100-01-01"))
    keep = joined["date"].ge(eligible_from) & joined["date"].le(eligible_to)
    retained = joined.loc[keep, [column for column in daily.columns if column in joined]].copy()
    excluded = joined.loc[~keep].copy()
    retained = retained.sort_values(["security_id", "date"], kind="stable").reset_index(
        drop=True
    )
    excluded = excluded.sort_values(
        ["security_id", "date"], kind="stable"
    ).reset_index(drop=True)
    if retained.duplicated(["security_id", "date"]).any():
        raise ValueError("Eligibility policy created duplicate security-date keys")
    retained_ids = set(retained["security_id"])
    if retained_ids != selected_ids:
        raise ValueError("Eligibility policy accidentally removed a security identity")
    return retained, excluded


def _q_gt(value: pd.Series, threshold: float) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").gt(threshold).fillna(False)


def recompute_diagnostics(
    raw: pd.DataFrame,
    calendar_sessions: pd.DatetimeIndex,
    *,
    return_threshold: float,
    volume_ratio_threshold: float,
    volume_lookback: int,
    volume_min_history: int,
) -> pd.DataFrame:
    """Recompute diagnostics within each retained security-history segment."""
    required = {
        "date",
        "security_id",
        "ticker",
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
        raise ValueError(f"Raw frame is missing columns: {missing}")
    if raw.duplicated(["security_id", "date"]).any():
        raise ValueError("Retained raw frame contains duplicate security-date keys")
    sessions = pd.DatetimeIndex(calendar_sessions).normalize()
    rows: list[pd.DataFrame] = []
    for security_id, group in raw.groupby("security_id", sort=False):
        clean = group.sort_values("date", kind="stable").copy()
        clean["date"] = pd.to_datetime(clean["date"]).dt.normalize()
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
        rolling_median = previous_volume.rolling(
            window=volume_lookback,
            min_periods=volume_min_history,
        ).median()
        extreme_volume = (
            rolling_median.gt(0)
            & clean["volume"].gt(rolling_median * volume_ratio_threshold)
        ).fillna(False)
        result = pd.DataFrame(
            {
                "date": clean["date"],
                "security_id": str(security_id),
                "ticker": clean["ticker"],
                "adj_close": clean["adj_close"],
                "diagnostic_pct_return": diagnostic_return,
                "extreme_return_flag": diagnostic_return.abs()
                .gt(return_threshold)
                .fillna(False),
                "price_discontinuity_flag": diagnostic_return.abs()
                .gt(return_threshold)
                .fillna(False),
                "raw_adjusted_return_conflict_flag": raw_adjusted_conflict.fillna(
                    False
                ),
                "ohlc_consistency_flag": (high_bad | low_bad).fillna(False),
                "zero_or_negative_adj_close_flag": clean["adj_close"]
                .le(0)
                .fillna(False),
                "zero_volume_flag": clean["volume"].eq(0).fillna(False),
                "negative_volume_flag": clean["volume"].lt(0).fillna(False),
                "extreme_volume_spike_flag": extreme_volume,
                "non_session_date_flag": ~clean["date"].isin(sessions),
                "stock_splits": clean["stock_splits"],
                "dividends": clean["dividends"],
            }
        )
        rows.append(result)
    diagnostics = pd.concat(rows, ignore_index=True)
    return diagnostics[DIAGNOSTIC_COLUMNS].sort_values(
        ["security_id", "date"], kind="stable"
    ).reset_index(drop=True)


def _strict_episodes(
    frame: pd.DataFrame,
    mask: pd.Series,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    flagged = frame.loc[mask, ["security_id", "date"]].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=["security_id", "date", "period_id", "start_date"]
        )
    return assign_episode_keys(flagged, sessions)


def _episode_metrics(
    diagnostics: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    zero_keys = _strict_episodes(
        diagnostics, diagnostics["zero_volume_flag"], sessions
    )
    volume_keys = _strict_episodes(
        diagnostics, diagnostics["extreme_volume_spike_flag"], sessions
    )
    return_keys = _strict_episodes(
        diagnostics, diagnostics["extreme_return_flag"], sessions
    )
    overlap_keys = _strict_episodes(
        diagnostics,
        diagnostics["extreme_volume_spike_flag"]
        | diagnostics["extreme_return_flag"],
        sessions,
    )
    if overlap_keys.empty:
        overlapping_episode_count = 0
    else:
        overlap = overlap_keys.merge(
            diagnostics[
                [
                    "security_id",
                    "date",
                    "extreme_volume_spike_flag",
                    "extreme_return_flag",
                ]
            ],
            on=["security_id", "date"],
            how="left",
            validate="many_to_one",
        )
        flags = overlap.groupby("period_id", sort=False).agg(
            has_volume=("extreme_volume_spike_flag", "any"),
            has_return=("extreme_return_flag", "any"),
        )
        overlapping_episode_count = int((flags["has_volume"] & flags["has_return"]).sum())
    return {
        "zero_keys": zero_keys,
        "zero_episodes": int(zero_keys["period_id"].nunique()),
        "zero_longest_episode": (
            int(
                zero_keys.groupby("period_id", sort=False)
                .size()
                .max()
            )
            if not zero_keys.empty
            else 0
        ),
        "volume_keys": volume_keys,
        "volume_episodes": int(volume_keys["period_id"].nunique()),
        "return_keys": return_keys,
        "return_episodes": int(return_keys["period_id"].nunique()),
        "overlapping_volume_return_episodes": overlapping_episode_count,
    }


def _coverage(
    daily: pd.DataFrame,
    master: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    observed = {
        security_id: pd.DatetimeIndex(group["date"].unique())
        for security_id, group in daily.groupby("security_id", sort=False)
    }
    rows = []
    for row in master.itertuples(index=False):
        dates = observed.get(row.security_id, pd.DatetimeIndex([]))
        first = dates.min() if len(dates) else pd.NaT
        last = dates.max() if len(dates) else pd.NaT
        internal = 0
        if len(dates):
            expected = sessions[(sessions >= first) & (sessions <= last)]
            internal = int(len(expected.difference(dates)))
        rows.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker_2023,
                "first_observation": first,
                "last_observation": last,
                "leading_xnys_sessions_not_retained": (
                    int((sessions < first).sum()) if pd.notna(first) else len(sessions)
                ),
                "trailing_xnys_sessions_not_retained": (
                    int((sessions > last).sum()) if pd.notna(last) else 0
                ),
                "internal_xnys_gaps": internal,
            }
        )
    return pd.DataFrame(rows)


def _remaining_zero_cases(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    zero = daily.loc[daily["volume"].eq(0)].copy()
    if zero.empty:
        return pd.DataFrame(
            columns=[
                "security_id",
                "ticker",
                "date",
                "zero_volume_episode_id",
                "episode_start",
                "episode_end",
                "episode_sessions",
                "zero_volume_classification",
            ]
        )
    episode_summary = (
        episodes.groupby("period_id", sort=False)
        .agg(
            episode_start=("date", "min"),
            episode_end=("date", "max"),
            episode_sessions=("date", "size"),
        )
        .reset_index()
    )
    zero = zero.merge(
        episodes[["security_id", "date", "period_id"]],
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
    ).merge(episode_summary, on="period_id", how="left", validate="many_to_one")
    zero = zero.rename(columns={"period_id": "zero_volume_episode_id"})
    zero["zero_volume_classification"] = "vendor_confirmed_market_unverified"
    return zero.sort_values(["security_id", "date"], kind="stable").reset_index(
        drop=True
    )


def _extreme_cases(
    raw: pd.DataFrame,
    diagnostics: pd.DataFrame,
    price_cases: pd.DataFrame,
    extreme_return_cases: pd.DataFrame,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    restored = raw.copy()
    restored["date"] = pd.to_datetime(restored["date"]).dt.normalize()
    restored = restored.sort_values(["security_id", "date"], kind="stable")
    factors: list[pd.Series] = []
    from src.investigation.anomaly_episodes import future_split_factor

    for _, group in restored.groupby("security_id", sort=False):
        split_factors = group["stock_splits"].where(
            group["date"].le(end_date), 1.0
        )
        factors.append(future_split_factor(split_factors))
    restored["future_split_factor"] = pd.concat(factors).sort_index()
    close = pd.to_numeric(restored["close"], errors="coerce")
    volume = pd.to_numeric(restored["volume"], errors="coerce")
    restored["daily_dollar_volume"] = np.where(
        close.gt(0).fillna(False)
        & volume.ge(0).fillna(False)
        & restored["future_split_factor"].gt(0).fillna(False),
        close * restored["future_split_factor"] * volume,
        np.nan,
    )

    price_evidence: dict[tuple[str, pd.Timestamp], str] = {}
    for row in price_cases.itertuples(index=False):
        if str(row.classification) == "not_classified_extreme":
            continue
        for field in ("min_date", "max_date"):
            value = getattr(row, field)
            if pd.notna(value):
                price_evidence[(str(row.security_id), pd.Timestamp(value))] = str(
                    row.classification
                )
    return_class: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for row in extreme_return_cases.itertuples(index=False):
        return_class[(str(row.security_id), pd.Timestamp(row.date))] = {
            "classification": str(row.classification),
            "uncertainty": str(row.uncertainty),
            "vendor_repaired": bool(row.vendor_repaired),
        }

    frame = diagnostics.merge(
        restored[
            [
                "security_id",
                "date",
                "close",
                "volume",
                "stock_splits",
                "dividends",
                "future_split_factor",
                "daily_dollar_volume",
            ]
        ],
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
    )
    keys = list(zip(frame["security_id"].astype(str), frame["date"], strict=True))
    frame["price_level_classification"] = [
        price_evidence.get(key, "") for key in keys
    ]
    frame["extreme_price_flag"] = (
        frame["zero_or_negative_adj_close_flag"]
        | frame["price_level_classification"].ne("")
    )
    mask = (
        frame["extreme_volume_spike_flag"]
        | frame["extreme_return_flag"]
        | frame["extreme_price_flag"]
    )
    cases = frame.loc[mask].copy()
    if cases.empty:
        return pd.DataFrame()
    cases["anomaly_types"] = cases.apply(
        lambda row: ";".join(
            label
            for label, flag in [
                ("extreme_volume", bool(row["extreme_volume_spike_flag"])),
                ("extreme_price", bool(row["extreme_price_flag"])),
                ("extreme_return", bool(row["extreme_return_flag"])),
            ]
            if flag
        ),
        axis=1,
    )
    # Build evidence by key to avoid positional assumptions after filtering.
    case_keys = list(
        zip(cases["security_id"].astype(str), cases["date"], strict=True)
    )
    cases["extreme_return_classification"] = [
        return_class.get(key, {}).get("classification", "") for key in case_keys
    ]
    cases["return_evidence_uncertainty"] = [
        return_class.get(key, {}).get("uncertainty", "") for key in case_keys
    ]
    cases["vendor_repair_flag"] = [
        bool(return_class.get(key, {}).get("vendor_repaired", False))
        for key in case_keys
    ]
    corporate_action_related = (
        cases["stock_splits"].gt(0).fillna(False)
        | cases["dividends"].gt(0).fillna(False)
        | cases["extreme_return_classification"].str.contains(
            "split|dividend|distribution", case=False, regex=True
        )
    )
    cases["corporate_action_related"] = corporate_action_related
    cases["unresolved_evidence"] = cases[
        "extreme_return_classification"
    ].str.contains("unresolved|near_zero|missing", case=False, regex=True)
    output_columns = [
        "security_id",
        "ticker",
        "date",
        "anomaly_types",
        "volume",
        "close",
        "adj_close",
        "diagnostic_pct_return",
        "daily_dollar_volume",
        "extreme_volume_spike_flag",
        "extreme_price_flag",
        "extreme_return_flag",
        "stock_splits",
        "dividends",
        "extreme_return_classification",
        "price_level_classification",
        "return_evidence_uncertainty",
        "vendor_repair_flag",
        "corporate_action_related",
        "unresolved_evidence",
    ]
    return cases[output_columns].sort_values(
        ["security_id", "date"], kind="stable"
    ).reset_index(drop=True)


def _rate_per_100k(value: float, observations: float) -> float:
    return 100_000.0 * value / observations if observations else np.nan


def _summary_table(
    *,
    acquired_ids: set[str],
    filtered_ids: set[str],
    final_ids: set[str],
    acquired_master: pd.DataFrame,
    selected_master: pd.DataFrame,
    acquired_base_rows: int,
    selected_rows: int,
    final_rows: int,
    acquired_diagnostics: pd.DataFrame,
    selected_diagnostics: pd.DataFrame,
    final_diagnostics: pd.DataFrame,
    ferg_excluded: int,
    dkng_excluded: int,
    acquired_internal_gaps: int,
    selected_internal_gaps: int,
    final_internal_gaps: int,
    acquired_price_cases: int,
    selected_price_cases: int,
    acquired_unresolved_returns: int,
    selected_unresolved_returns: int,
    final_unresolved_returns: int,
    acquired_corporate_cases: int,
    selected_corporate_cases: int,
    final_corporate_cases: int,
    acquired_quarantine: int,
    selected_quarantine: int,
    final_quarantine: int,
) -> pd.DataFrame:
    scopes = {
        "original_acquired": {
            "rows": acquired_base_rows,
            "securities": len(acquired_ids),
            "issuers": acquired_master["cik"].nunique(),
            "diagnostics": acquired_diagnostics,
        },
        "original_liquidity_filtered": {
            "rows": selected_rows,
            "securities": len(filtered_ids),
            "issuers": selected_master["cik"].nunique(),
            "diagnostics": selected_diagnostics,
        },
        "final_research_dataset": {
            "rows": final_rows,
            "securities": len(final_ids),
            "issuers": selected_master["cik"].nunique(),
            "diagnostics": final_diagnostics,
        },
    }

    def counts(diagnostics: pd.DataFrame) -> dict[str, int]:
        return {
            "zero_volume": int(diagnostics["zero_volume_flag"].sum()),
            "extreme_volume": int(diagnostics["extreme_volume_spike_flag"].sum()),
            "extreme_return": int(diagnostics["extreme_return_flag"].sum()),
            "nonpositive_adjusted_price": int(
                diagnostics["zero_or_negative_adj_close_flag"].sum()
            ),
        }

    c = {name: counts(scope["diagnostics"]) for name, scope in scopes.items()}
    rows = []
    metric_specs = [
        ("Securities", [len(acquired_ids), len(filtered_ids), len(final_ids)]),
        (
            "Distinct SEC issuers",
            [
                acquired_master["cik"].nunique(),
                selected_master["cik"].nunique(),
                selected_master["cik"].nunique(),
            ],
        ),
        ("Validated stock-day observations", [acquired_base_rows, selected_rows, final_rows]),
        ("FERG observations excluded", [0, 0, ferg_excluded]),
        ("DKNG observations excluded", [0, 0, dkng_excluded]),
        ("Missing adjusted-close observations", [0, 0, 0]),
        ("Missing-volume observations", [0, 0, 0]),
        (
            "Zero-volume observations",
            [
                c["original_acquired"]["zero_volume"],
                c["original_liquidity_filtered"]["zero_volume"],
                c["final_research_dataset"]["zero_volume"],
            ],
        ),
        (
            "Extreme-volume observations",
            [
                c["original_acquired"]["extreme_volume"],
                c["original_liquidity_filtered"]["extreme_volume"],
                c["final_research_dataset"]["extreme_volume"],
            ],
        ),
        (
            "Extreme diagnostic-return observations",
            [
                c["original_acquired"]["extreme_return"],
                c["original_liquidity_filtered"]["extreme_return"],
                c["final_research_dataset"]["extreme_return"],
            ],
        ),
        (
            "Nonpositive adjusted-price observations",
            [
                c["original_acquired"]["nonpositive_adjusted_price"],
                c["original_liquidity_filtered"]["nonpositive_adjusted_price"],
                c["final_research_dataset"]["nonpositive_adjusted_price"],
            ],
        ),
        (
            "Existing price-level case records",
            [acquired_price_cases, selected_price_cases, selected_price_cases],
        ),
        (
            "Internal missing XNYS sessions",
            [acquired_internal_gaps, selected_internal_gaps, final_internal_gaps],
        ),
        (
            "Unresolved extreme-return discontinuities",
            [
                acquired_unresolved_returns,
                selected_unresolved_returns,
                final_unresolved_returns,
            ],
        ),
        (
            "Corporate-action discrepancy records",
            [
                acquired_corporate_cases,
                selected_corporate_cases,
                final_corporate_cases,
            ],
        ),
        (
            "Quarantine-candidate rows/windows",
            [acquired_quarantine, selected_quarantine, final_quarantine],
        ),
    ]
    denominators = [acquired_base_rows, selected_rows, final_rows]
    for metric, values in metric_specs:
        rates = [
            _rate_per_100k(value, denominator)
            if metric not in {
                "Securities",
                "Distinct SEC issuers",
                "Validated stock-day observations",
                "FERG observations excluded",
                "DKNG observations excluded",
            }
            else np.nan
            for value, denominator in zip(values, denominators, strict=True)
        ]
        rows.append(
            {
                "metric": metric,
                "original_acquired": values[0],
                "original_liquidity_filtered": values[1],
                "final_research_dataset": values[2],
                "original_acquired_per_100k": rates[0],
                "original_liquidity_filtered_per_100k": rates[1],
                "final_research_dataset_per_100k": rates[2],
            }
        )
    return pd.DataFrame(rows)


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(project_path(config_path))
    input_cfg = config["inputs"]
    output_root = project_path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    policy = load_eligibility_policy(project_path(input_cfg["policy"]))

    original_master_path = project_path(input_cfg["original_selected_master"])
    original_daily_path = project_path(input_cfg["original_selected_daily"])
    raw_path = project_path(input_cfg["standardized_raw"])
    diagnostic_path = project_path(input_cfg["original_diagnostic_returns"])
    policy_path = project_path(input_cfg["policy"])
    evidence_path = project_path(input_cfg["eligibility_evidence"])
    manifest_path = project_path(input_cfg["original_download_manifest"])
    universe_master_path = project_path(input_cfg["original_universe_master"])
    quality_by_security_path = project_path(
        input_cfg["original_quality_by_security"]
    )
    original_files = [
        original_master_path,
        original_daily_path,
        raw_path,
        diagnostic_path,
        policy_path,
        evidence_path,
        manifest_path,
        universe_master_path,
        quality_by_security_path,
    ]
    before_hashes = {str(path): _sha256(path) for path in original_files}
    raw_root = project_path(input_cfg["raw_yahoo_dir"])
    before_raw_tree = tree_sha256(raw_root) if raw_root.exists() else None
    filtered_root = project_path(input_cfg["original_filter_root"])
    anomaly_dir = project_path(input_cfg["original_anomaly_dir"])
    before_filtered_tree = (
        tree_sha256(filtered_root, pattern="*") if filtered_root.exists() else None
    )
    before_anomaly_tree = (
        tree_sha256(anomaly_dir, pattern="*") if anomaly_dir.exists() else None
    )

    original_master = pd.read_parquet(original_master_path)
    original_daily = pd.read_parquet(original_daily_path)
    selected_ids = set(original_master["security_id"].astype(str))
    original_daily["security_id"] = original_daily["security_id"].astype(str)
    final_daily, excluded = apply_eligibility_policy(original_daily, policy)
    final_daily = final_daily[DAILY_COLUMNS].copy()
    final_daily["pct_return"] = np.nan
    final_master = original_master.copy()
    policy_lookup = policy.set_index("security_id")
    final_master["historical_eligibility_status"] = final_master[
        "security_id"
    ].astype(str).map(
        lambda value: (
            "historical_boundary_applied"
            if value in policy_lookup.index
            else "default_2016_2025_history"
        )
    )
    final_master["historical_eligible_from"] = final_master[
        "security_id"
    ].astype(str).map(policy_lookup["eligible_from"])
    final_master["historical_eligible_to"] = final_master[
        "security_id"
    ].astype(str).map(policy_lookup["eligible_to"])
    final_master["historical_policy_type"] = final_master[
        "security_id"
    ].astype(str).map(policy_lookup["policy_type"])
    final_master["historical_policy_reference"] = final_master[
        "security_id"
    ].astype(str).map(policy_lookup["source_reference"])

    calendar = xcals.get_calendar(config["project"]["calendar"])
    sessions = calendar.sessions_in_range(
        config["project"]["start_date"], config["project"]["end_date"]
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
            "source_file",
        ],
        filters=[("security_id", "in", sorted(selected_ids))],
    )
    raw["security_id"] = raw["security_id"].astype(str)
    retained_raw, _ = apply_eligibility_policy(
        raw.rename(columns={"ticker_2023": "ticker"}),
        policy,
    )
    diagnostics = recompute_diagnostics(
        retained_raw,
        sessions,
        return_threshold=float(config["project"]["extreme_return_abs_threshold"]),
        volume_ratio_threshold=float(
            config["project"]["extreme_volume_ratio_threshold"]
        ),
        volume_lookback=int(config["project"]["extreme_volume_lookback"]),
        volume_min_history=int(config["project"]["extreme_volume_min_history"]),
    )

    source_map = raw[
        ["security_id", "date", "source_file"]
    ].drop_duplicates(["security_id", "date"])
    exclusions = excluded.merge(
        source_map,
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
    )
    exclusions["policy_reference"] = (
        exclusions["policy_type"].astype(str)
        + ":"
        + exclusions["eligible_from"].dt.date.astype(str)
    )
    exclusions = exclusions.rename(
        columns={"source_file": "original_source_file"}
    )
    exclusion_columns = [
        "security_id",
        "ticker",
        "date",
        "exclusion_reason",
        "policy_reference",
        "original_source_file",
    ]
    metrics = _episode_metrics(diagnostics, sessions)
    coverage = _coverage(final_daily, final_master, sessions)
    zero_cases = _remaining_zero_cases(final_daily, metrics["zero_keys"])
    quality_dir = project_path(input_cfg["original_quality_dir"])
    price_cases = pd.read_csv(
        quality_dir / "filtered_extreme_price_cases.csv", low_memory=False
    )
    extreme_return_cases = pd.read_csv(
        quality_dir / "filtered_extreme_return_cases.csv", low_memory=False
    )
    extreme_cases = _extreme_cases(
        retained_raw,
        diagnostics,
        price_cases,
        extreme_return_cases,
        pd.Timestamp(config["project"]["end_date"]),
    )

    final_keys = set(
        zip(
            final_daily["security_id"].astype(str),
            pd.to_datetime(final_daily["date"]),
            strict=True,
        )
    )
    # No return diagnostic is carried across an eligibility boundary.
    filtered_extremes = extreme_return_cases.assign(
        date=pd.to_datetime(extreme_return_cases["date"])
    ).loc[
        [
            (str(security_id), date) in final_keys
            for security_id, date in zip(
                extreme_return_cases["security_id"].astype(str),
                pd.to_datetime(extreme_return_cases["date"]),
                strict=True,
            )
        ]
    ]

    acquired_manifest = pd.read_csv(manifest_path, low_memory=False)
    acquired_ids = set(
        acquired_manifest.loc[
            acquired_manifest["download_status"].eq("success"), "security_id"
        ].astype(str)
    )
    universe_master = pd.read_parquet(universe_master_path)
    acquired_master = universe_master[
        universe_master["security_id"].astype(str).isin(acquired_ids)
    ]
    acquired_base_rows = len(
        pd.read_parquet(project_path(input_cfg["acquired_base"]), columns=["date"])
    )
    acquired_diag = pd.read_parquet(diagnostic_path)
    selected_diag = acquired_diag[
        acquired_diag["security_id"].astype(str).isin(selected_ids)
    ]
    acquired_diag = acquired_diag[
        acquired_diag["security_id"].astype(str).isin(acquired_ids)
    ]
    original_quality = pd.read_csv(quality_by_security_path, low_memory=False)
    acquired_quality = original_quality[
        original_quality["security_id"].astype(str).isin(acquired_ids)
    ]
    selected_quality = original_quality[
        original_quality["security_id"].astype(str).isin(selected_ids)
    ]
    all_price_cases = pd.read_csv(
        anomaly_dir / "extreme_price_cases.csv", low_memory=False
    )
    all_extreme_cases = pd.read_csv(
        anomaly_dir / "extreme_return_cases.csv", low_memory=False
    )
    selected_extreme = all_extreme_cases[
        all_extreme_cases["security_id"].astype(str).isin(selected_ids)
    ]
    all_quarantine = pd.read_csv(
        anomaly_dir / "analysis_quarantine_candidates.csv", low_memory=False
    )
    selected_quarantine = all_quarantine[
        all_quarantine["security_id"].astype(str).isin(selected_ids)
    ]
    split_recon = pd.read_csv(
        anomaly_dir / "corporate_action_split_reconciliation.csv", low_memory=False
    )
    dividend_recon = pd.read_csv(
        anomaly_dir / "corporate_action_dividend_reconciliation.csv",
        low_memory=False,
    )
    selected_split_recon = pd.read_csv(
        quality_dir / "filtered_split_reconciliation.csv", low_memory=False
    )
    selected_dividend_recon = pd.read_csv(
        quality_dir / "filtered_dividend_reconciliation.csv", low_memory=False
    )
    corp_all = len(split_recon) + len(dividend_recon)
    corp_selected = len(selected_split_recon) + len(selected_dividend_recon)

    summary = _summary_table(
        acquired_ids=acquired_ids,
        filtered_ids=selected_ids,
        final_ids=set(final_daily["security_id"].astype(str)),
        acquired_master=acquired_master,
        selected_master=final_master,
        acquired_base_rows=acquired_base_rows,
        selected_rows=len(original_daily),
        final_rows=len(final_daily),
        acquired_diagnostics=acquired_diag,
        selected_diagnostics=selected_diag,
        final_diagnostics=diagnostics,
        ferg_excluded=int(exclusions["ticker"].eq("FERG").sum()),
        dkng_excluded=int(exclusions["ticker"].eq("DKNG").sum()),
        acquired_internal_gaps=int(
            acquired_quality["unobserved_internal_sessions"].sum()
        ),
        selected_internal_gaps=int(
            selected_quality["unobserved_internal_sessions"].sum()
        ),
        final_internal_gaps=int(coverage["internal_xnys_gaps"].sum()),
        acquired_price_cases=len(all_price_cases),
        selected_price_cases=len(price_cases),
        acquired_unresolved_returns=int(
            all_extreme_cases["classification"].eq(
                "unresolved_extreme_discontinuity"
            ).sum()
        ),
        selected_unresolved_returns=int(
            selected_extreme["classification"].eq(
                "unresolved_extreme_discontinuity"
            ).sum()
        ),
        final_unresolved_returns=int(
            filtered_extremes["classification"].eq(
                "unresolved_extreme_discontinuity"
            ).sum()
        ),
        acquired_corporate_cases=corp_all,
        selected_corporate_cases=corp_selected,
        final_corporate_cases=corp_selected,
        acquired_quarantine=len(all_quarantine),
        selected_quarantine=len(selected_quarantine),
        final_quarantine=len(selected_quarantine),
    )

    zero_by_security = (
        zero_cases.groupby(["security_id", "ticker"], sort=True)
        .agg(
            zero_volume_rows=("date", "size"),
            first_zero_date=("date", "min"),
            last_zero_date=("date", "max"),
            zero_volume_episodes=("zero_volume_episode_id", "nunique"),
        )
        .reset_index()
        if not zero_cases.empty
        else pd.DataFrame(
            columns=[
                "security_id",
                "ticker",
                "zero_volume_rows",
                "first_zero_date",
                "last_zero_date",
                "zero_volume_episodes",
            ]
        )
    )
    zero_by_security["classification"] = "vendor_confirmed_market_unverified"

    _atomic_parquet(final_daily, output_root / "selected_equity_daily.parquet")
    _atomic_parquet(final_master, output_root / "selected_security_master.parquet")
    _atomic_parquet(diagnostics, output_root / "diagnostic_returns.parquet")
    _atomic_csv(exclusions[exclusion_columns], output_root / "historical_exclusions.csv")
    _atomic_csv(summary, output_root / "final_data_quality_summary.csv")
    _atomic_csv(zero_cases, output_root / "remaining_zero_volume_cases.csv")
    _atomic_csv(zero_by_security, output_root / "remaining_zero_volume_by_security.csv")
    _atomic_csv(extreme_cases, output_root / "remaining_extreme_value_cases.csv")
    _atomic_csv(coverage, output_root / "security_coverage.csv")

    after_hashes = {str(path): _sha256(path) for path in original_files}
    after_raw_tree = tree_sha256(raw_root) if raw_root.exists() else None
    after_filtered_tree = (
        tree_sha256(filtered_root, pattern="*") if filtered_root.exists() else None
    )
    after_anomaly_tree = (
        tree_sha256(anomaly_dir, pattern="*") if anomaly_dir.exists() else None
    )
    if (
        before_hashes != after_hashes
        or before_raw_tree != after_raw_tree
        or before_filtered_tree != after_filtered_tree
        or before_anomaly_tree != after_anomaly_tree
    ):
        raise RuntimeError("A protected source file changed during eligibility build")
    if final_daily.duplicated(["security_id", "date"]).any():
        raise RuntimeError("Final research panel contains duplicate keys")
    if set(final_daily["security_id"].astype(str)) != selected_ids:
        raise RuntimeError("Fixed 488-security membership changed")
    if not final_daily["pct_return"].isna().all():
        raise RuntimeError("Final pct_return field must remain null")
    if int(exclusions["ticker"].eq("FERG").sum()) != 1601:
        raise RuntimeError("FERG exclusion count mismatch")
    if int(exclusions["ticker"].eq("DKNG").sum()) != 189:
        raise RuntimeError("DKNG exclusion count mismatch")
    if int(exclusions["security_id"].nunique()) != len(policy):
        raise RuntimeError("Eligibility policy affected a security outside its rows")

    final_volume_count = int(diagnostics["extreme_volume_spike_flag"].sum())
    final_return_count = int(diagnostics["extreme_return_flag"].sum())
    final_price_count = int(extreme_cases["extreme_price_flag"].sum()) if not extreme_cases.empty else 0
    zero_rows = int(diagnostics["zero_volume_flag"].sum())
    zero_securities = int(
        diagnostics.loc[diagnostics["zero_volume_flag"], "security_id"].nunique()
    )
    final_summary = {
        "generated_at_utc": utc_now(),
        "snapshot": {
            "securities": int(final_master["security_id"].nunique()),
            "issuers": int(final_master["cik"].nunique()),
            "observations": int(len(final_daily)),
            "diagnostic_rows": int(len(diagnostics)),
            "ferg_excluded": int(exclusions["ticker"].eq("FERG").sum()),
            "dkng_excluded": int(exclusions["ticker"].eq("DKNG").sum()),
            "total_excluded": int(len(exclusions)),
            "zero_volume_rows": zero_rows,
            "zero_volume_securities": zero_securities,
            "zero_volume_episodes": metrics["zero_episodes"],
            "longest_zero_volume_episode": metrics["zero_longest_episode"],
            "extreme_volume_rows": final_volume_count,
            "extreme_volume_securities": int(
                diagnostics.loc[
                    diagnostics["extreme_volume_spike_flag"], "security_id"
                ].nunique()
            ),
            "extreme_volume_episodes": metrics["volume_episodes"],
            "extreme_volume_max_share_volume": (
                float(
                    extreme_cases.loc[
                        extreme_cases["extreme_volume_spike_flag"], "volume"
                    ].max()
                )
                if final_volume_count
                else 0.0
            ),
            "extreme_volume_max_daily_dollar_volume": (
                float(
                    extreme_cases.loc[
                        extreme_cases["extreme_volume_spike_flag"],
                        "daily_dollar_volume",
                    ].max()
                )
                if final_volume_count
                else 0.0
            ),
            "extreme_price_rows": final_price_count,
            "extreme_price_securities": int(
                extreme_cases.loc[
                    extreme_cases["extreme_price_flag"], "security_id"
                ].nunique()
            )
            if final_price_count
            else 0,
            "extreme_return_rows": final_return_count,
            "extreme_return_securities": int(
                diagnostics.loc[
                    diagnostics["extreme_return_flag"], "security_id"
                ].nunique()
            ),
            "extreme_return_positive": int(
                (
                    diagnostics["extreme_return_flag"]
                    & diagnostics["diagnostic_pct_return"].gt(0)
                ).sum()
            ),
            "extreme_return_negative": int(
                (
                    diagnostics["extreme_return_flag"]
                    & diagnostics["diagnostic_pct_return"].lt(0)
                ).sum()
            ),
            "extreme_return_episodes": metrics["return_episodes"],
            "overlapping_extreme_volume_return_episodes": metrics[
                "overlapping_volume_return_episodes"
            ],
            "corporate_action_related_extreme_returns": int(
                (
                    extreme_cases["extreme_return_flag"]
                    & extreme_cases["corporate_action_related"]
                ).sum()
            ),
            "unresolved_extreme_returns": int(
                filtered_extremes["classification"].eq(
                    "unresolved_extreme_discontinuity"
                ).sum()
            ),
            "remaining_unresolved_quality_cases": {
                "zero_volume_cause_unverified": zero_rows,
                "ordinary_dividend_adjustment_mismatch": int(
                    selected_dividend_recon["final_classification"].eq(
                        "ordinary_dividend_adjustment_mismatch"
                    ).sum()
                ),
                "ohlc_consistency_open": int(
                    diagnostics["ohlc_consistency_flag"].sum()
                ),
                "identity_review_candidates": 2,
            },
            "fixed_membership_unchanged": True,
        },
        "policy": policy.assign(
            eligible_from=policy["eligible_from"].dt.date.astype(str),
            eligible_to=policy["eligible_to"].map(
                lambda value: "" if pd.isna(value) else value.date().isoformat()
            ),
        ).to_dict("records"),
        "eligibility_evidence": json.loads(
            evidence_path.read_text(encoding="utf-8")
        ),
        "research_constraints": config["research_constraints"],
        "protected_input_hashes_sha256": after_hashes,
        "protected_directory_hashes_sha256": {
            "raw_yahoo": after_raw_tree,
            "original_liquidity_filter": after_filtered_tree,
            "original_anomaly_investigation": after_anomaly_tree,
        },
        "raw_yahoo_tree_sha256": after_raw_tree,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "exchange_calendars": getattr(xcals, "__version__", "unknown"),
        },
    }
    final_summary["treatments"] = [
        {
            "issue": "FERG before 2022-05-12",
            "final_treatment": "exclude_from_research_dataset",
        },
        {
            "issue": "DKNG before 2020-04-24",
            "final_treatment": "exclude_from_research_dataset",
        },
        {
            "issue": "Remaining Yahoo zero-volume observations",
            "final_treatment": "retain_with_vendor_confirmed_market_unverified_flag",
        },
        {
            "issue": "Extreme trading volume",
            "final_treatment": "retain_unless_confirmed_erroneous",
        },
        {
            "issue": "Extreme adjusted prices",
            "final_treatment": "retain_or_flag_using_existing_evidence",
        },
        {
            "issue": "Extreme diagnostic returns",
            "final_treatment": "retain_unless_confirmed_erroneous",
        },
        {
            "issue": "Unresolved non-volume defects",
            "final_treatment": "preserve_flags_and_document",
        },
    ]
    atomic_write_json(final_summary, output_root / "cleaning_policy.json")
    atomic_write_json(
        {
            "status": "complete",
            "snapshot": final_summary["snapshot"],
            "protected_inputs_unchanged": True,
            "all_policy_boundaries_inclusive": True,
        },
        output_root / "execution_summary.json",
    )
    LOGGER.info(
        "Built final panel: %d securities, %d observations, %d excluded",
        final_summary["snapshot"]["securities"],
        final_summary["snapshot"]["observations"],
        final_summary["snapshot"]["total_excluded"],
    )
    return final_summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "historical_eligibility.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
