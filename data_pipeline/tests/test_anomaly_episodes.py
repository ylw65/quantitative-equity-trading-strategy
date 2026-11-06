from __future__ import annotations

import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT
from src.investigation.anomaly_episodes import (
    ANOMALY_FIELDS,
    DAILY_COLUMNS,
    PERIOD_COLUMNS,
    REVIEW_COLUMNS,
    assign_episode_keys,
    assign_review_priority,
    future_split_factor,
    restore_compatible_close,
    stable_period_id,
)


OUTPUT_ROOT = PROJECT_ROOT / "data/investigation/filtered_anomaly_episodes"


def _calendar() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def test_grouping_skips_weekends_and_exchange_holidays() -> None:
    calendar = _calendar()
    sessions = calendar.sessions_in_range("2022-12-19", "2023-01-03")
    flagged = pd.DataFrame(
        {
            "security_id": ["SEC:A"] * 3,
            "date": pd.to_datetime(
                ["2022-12-23", "2022-12-27", "2022-12-29"]
            ),
        }
    )
    result = assign_episode_keys(flagged, sessions)
    assert result["episode_number"].nunique() == 2
    assert result.iloc[0]["period_id"] == result.iloc[1]["period_id"]
    assert result.iloc[1]["period_id"] != result.iloc[2]["period_id"]


def test_intervening_unflagged_session_breaks_episode() -> None:
    calendar = _calendar()
    sessions = calendar.sessions_in_range("2022-12-19", "2023-01-03")
    flagged = pd.DataFrame(
        {
            "security_id": ["SEC:A", "SEC:A"],
            "date": pd.to_datetime(["2022-12-22", "2022-12-27"]),
        }
    )
    result = assign_episode_keys(flagged, sessions)
    assert result["episode_number"].nunique() == 2
    assert result["period_id"].nunique() == 2


def test_episodes_do_not_join_across_securities() -> None:
    calendar = _calendar()
    sessions = calendar.sessions_in_range("2022-12-19", "2022-12-30")
    flagged = pd.DataFrame(
        {
            "security_id": ["SEC:A", "SEC:B"],
            "date": pd.to_datetime(["2022-12-23", "2022-12-27"]),
        }
    )
    result = assign_episode_keys(flagged, sessions)
    assert result["period_id"].nunique() == 2
    assert result["episode_number"].nunique() == 2


def test_period_id_is_stable_and_security_specific() -> None:
    start = pd.Timestamp("2022-12-27")
    first = stable_period_id("SEC:A", start)
    assert first == stable_period_id("SEC:A", start)
    assert first != stable_period_id("SEC:B", start)
    assert first.startswith("AE-20221227-")


def test_future_split_factor_excludes_event_date() -> None:
    factors = future_split_factor(pd.Series([0.0, 2.0, 0.0, 3.0]))
    assert factors.tolist() == [6.0, 3.0, 3.0, 1.0]


def test_restored_close_and_dollar_volume_use_compatible_basis() -> None:
    raw = pd.DataFrame(
        {
            "security_id": ["SEC:A", "SEC:A", "SEC:A"],
            "date": pd.to_datetime(
                ["2022-12-27", "2022-12-28", "2022-12-29"]
            ),
            "close": [10.0, 5.0, 6.0],
            "volume": [100.0, 0.0, 200.0],
            "stock_splits": [0.0, 2.0, 0.0],
        }
    )
    result = restore_compatible_close(
        raw, end_date=pd.Timestamp("2025-12-31")
    )
    assert result["future_split_factor"].tolist() == [2.0, 1.0, 1.0]
    assert result["restored_close"].tolist() == [20.0, 5.0, 6.0]
    assert result["daily_dollar_volume"].tolist() == [2000.0, 0.0, 1200.0]
    assert not result["daily_dollar_volume"].isna().any()


def test_missing_price_and_volume_remain_null() -> None:
    raw = pd.DataFrame(
        {
            "security_id": ["SEC:A", "SEC:A"],
            "date": pd.to_datetime(["2022-12-27", "2022-12-28"]),
            "close": [10.0, np.nan],
            "volume": [np.nan, 100.0],
            "stock_splits": [0.0, 0.0],
        }
    )
    result = restore_compatible_close(
        raw, end_date=pd.Timestamp("2025-12-31")
    )
    assert result["daily_dollar_volume"].isna().all()


def test_review_priority_places_liquidity_window_first() -> None:
    period = pd.Series(
        {
            "n_zero_volume_sessions": 20,
            "n_extreme_volume_sessions": 1,
            "max_daily_dollar_volume": 2_000_000_000,
        }
    )
    internal = pd.Series(
        {
            "touches_liquidity_window": True,
            "has_unresolved_quality": True,
            "has_multiple_anomaly_types": True,
            "has_corporate_action": True,
            "all_extreme_returns_retained_market_moves": False,
        }
    )
    assert (
        assign_review_priority(
            period,
            internal,
            material_daily_dollar_volume_usd=100_000_000,
            multi_session_zero_volume_min_sessions=2,
        )
        == "P0_liquidity_window"
    )


def test_material_extreme_volume_is_high_priority() -> None:
    period = pd.Series(
        {
            "n_zero_volume_sessions": 0,
            "n_extreme_volume_sessions": 1,
            "max_daily_dollar_volume": 200_000_000,
        }
    )
    internal = pd.Series(
        {
            "touches_liquidity_window": False,
            "has_unresolved_quality": False,
            "has_multiple_anomaly_types": False,
            "has_corporate_action": False,
            "all_extreme_returns_retained_market_moves": False,
        }
    )
    assert (
        assign_review_priority(
            period,
            internal,
            material_daily_dollar_volume_usd=100_000_000,
            multi_session_zero_volume_min_sessions=2,
        )
        == "P1_material_extreme_volume"
    )


def test_output_files_and_reconciliations() -> None:
    required = [
        "anomaly_periods.csv",
        "anomaly_daily_observations.csv",
        "anomaly_episode_summary.csv",
        "anomaly_review_register.csv",
        "anomaly_episode_checks.csv",
        "execution_summary.json",
        "README.md",
    ]
    for name in required:
        assert (OUTPUT_ROOT / name).exists(), name

    # The CSV may have been saved by a spreadsheet tool using a legacy single-byte
    # encoding; all test-relevant counts remain numeric under that representation.
    periods = pd.read_csv(OUTPUT_ROOT / "anomaly_periods.csv", encoding="latin-1")
    daily = pd.read_csv(
        OUTPUT_ROOT / "anomaly_daily_observations.csv", low_memory=False
    )
    summary = pd.read_csv(OUTPUT_ROOT / "anomaly_episode_summary.csv")
    review = pd.read_csv(
        OUTPUT_ROOT / "anomaly_review_register.csv", low_memory=False
    )
    checks = pd.read_csv(OUTPUT_ROOT / "anomaly_episode_checks.csv")
    execution = json.loads(
        (OUTPUT_ROOT / "execution_summary.json").read_text(encoding="utf-8")
    )

    assert list(periods.columns) == PERIOD_COLUMNS
    assert set(DAILY_COLUMNS).issubset(daily.columns)
    assert list(review.columns) == REVIEW_COLUMNS
    assert periods["period_id"].is_unique
    assert not daily.duplicated(["period_id", "date"]).any()
    assert set(daily["period_id"]) == set(periods["period_id"])
    assert int(periods["n_zero_volume_sessions"].sum()) == 900
    assert int(periods["n_extreme_volume_sessions"].sum()) == 282
    assert int(periods["n_extreme_return_sessions"].sum()) == 45
    assert int(periods["n_extreme_price_sessions"].sum()) == 0

    category = summary[summary["metric"].eq("episodes_by_anomaly_type")].set_index(
        "category"
    )
    assert set(ANOMALY_FIELDS).issubset(category.index)
    assert category.loc["zero_volume", "flagged_observations"] == 900
    assert category.loc["extreme_volume", "flagged_observations"] == 282
    assert category.loc["extreme_return", "flagged_observations"] == 45
    assert category.loc["corporate_action_discrepancy", "flagged_observations"] == 4
    assert review["review_status"].eq("pending_review").all()
    assert review["reviewer_conclusion"].isna().all()
    assert review["recommended_treatment"].isna().all()
    assert checks["passed"].all()
    assert execution["all_checks_passed"]
    assert execution["raw_yahoo_tree_unchanged"]
    assert execution["snapshot"]["flagged_stock_days"] == 1212
