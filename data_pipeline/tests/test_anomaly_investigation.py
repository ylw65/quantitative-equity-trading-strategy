from __future__ import annotations

from src.investigation.anomaly_analysis import (
    classify_corporate_action,
    classify_extreme_return,
    classify_price_level,
)


def test_near_zero_denominator_precedes_action_context() -> None:
    assert (
        classify_extreme_return(
            0.005,
            100.0,
            same_day_split=True,
            same_day_dividend=False,
            split_within_five_sessions=True,
            vendor_repaired=False,
        )
        == "near_zero_denominator"
    )


def test_combined_split_and_distribution_is_classified() -> None:
    result = classify_corporate_action(
        event_kind="split",
        dividend_fraction=0.30,
        split_same_day=True,
    )
    assert result[0] == "simultaneous_split_and_distribution"
    assert result[1] == "resolved_methodology"


def test_large_dividend_is_not_called_ordinary_cash() -> None:
    result = classify_corporate_action(
        event_kind="dividend",
        dividend_fraction=0.50,
        split_same_day=False,
    )
    assert result[0] == "probable_special_distribution_or_spinoff"
    assert result[1] == "partially_resolved"


def test_price_level_classification_uses_split_context() -> None:
    assert (
        classify_price_level(1e13, 100.0, 1e8, reverse_split_count=12)
        == "consistent_with_reverse_split_adjustment_history"
    )
    assert (
        classify_price_level(2e13, 100.0, 1e8, reverse_split_count=0)
        == "unresolved_extreme_adjusted_price_level"
    )
