from __future__ import annotations

import numpy as np
import pandas as pd

from src.filtering.post_filter_integrity import (
    _raw_metrics,
    calendar_gap_intervals,
    identity_review_candidates,
    missing_intervals,
)


def test_missing_intervals_distinguish_internal_and_edges() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-01-04",
                    "2016-01-05",
                    "2016-01-06",
                    "2016-01-07",
                    "2016-01-08",
                    "2016-01-11",
                ]
            ),
            "adj_close": [np.nan, 10.0, np.nan, np.nan, 11.0, np.nan],
        }
    )
    intervals = missing_intervals(
        frame, "adj_close", security_id="SEC:1", ticker="AAA"
    )
    assert [row["interval_class"] for row in intervals] == [
        "leading_unknown_placeholder_or_pre_history",
        "internal",
        "trailing_unknown_placeholder_or_post_history",
    ]


def test_calendar_gap_intervals_use_only_internal_sessions() -> None:
    sessions = pd.DatetimeIndex(
        [
            "2022-12-27",
            "2022-12-28",
            "2022-12-29",
            "2022-12-30",
        ]
    )
    observed = pd.DatetimeIndex(["2022-12-27", "2022-12-30"])
    intervals = calendar_gap_intervals(
        observed,
        sessions,
        security_id="SEC:1",
        ticker="AAA",
    )
    assert len(intervals) == 1
    assert intervals[0]["session_count"] == 2
    assert intervals[0]["interval_class"] == "internal_calendar_gap"


def test_raw_metrics_scope_and_zero_volume_are_preserved() -> None:
    raw = pd.DataFrame(
        {
            "security_id": ["A", "A", "B"],
            "date": pd.to_datetime(["2022-12-29", "2022-12-30", "2022-12-30"]),
            "ticker": ["AAA", "AAA", "BBB"],
            "open": [1.0, np.nan, 2.0],
            "high": [1.1, np.nan, 2.1],
            "low": [0.9, np.nan, 1.9],
            "close": [1.0, np.nan, 2.0],
            "adj_close": [1.0, np.nan, 2.0],
            "volume": [0.0, np.nan, 10.0],
        }
    )
    base = pd.DataFrame(
        {
            "security_id": ["A", "B"],
            "date": pd.to_datetime(["2022-12-29", "2022-12-30"]),
        }
    )
    metrics = _raw_metrics(raw, base, {"A"})
    assert metrics["raw_rows"] == 2
    assert metrics["validated_rows"] == 1
    assert metrics["placeholder_all_nan_rows"] == 1
    assert metrics["zero_volume_rows"] == 1
    assert metrics["raw_rows_not_in_base"] == 1


def test_duplicate_key_detection() -> None:
    raw = pd.DataFrame(
        {
            "security_id": ["A", "A"],
            "date": pd.to_datetime(["2022-12-30", "2022-12-30"]),
            "ticker": ["AAA", "AAA"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "adj_close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    base = raw[["security_id", "date"]].drop_duplicates()
    metrics = _raw_metrics(raw, base, {"A"})
    assert metrics["duplicate_security_date_rows"] == 1


def test_identity_actions_are_review_candidates_not_contamination() -> None:
    identity = pd.DataFrame(
        {
            "security_id": ["A", "B", "C", "D"],
            "ticker_2023": ["AAA", "BBB", "CCC", "DDD"],
            "quarantine_action": ["none", "base_rows_only", None, "quarantine"],
        }
    )
    review = identity_review_candidates(identity, {"A", "B", "C"})
    assert review["security_id"].tolist() == ["B"]
    assert not review["quarantine_action"].eq("contamination").any()
