from __future__ import annotations

import json
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from src.common import PROJECT_ROOT
from src.filtering.historical_eligibility import (
    apply_eligibility_policy,
    recompute_diagnostics,
)


OUTPUT_ROOT = PROJECT_ROOT / "data/filtered/adtv100m_2023_eligibility_v2"


def _policy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": ["SEC:A"],
            "ticker": ["AAA"],
            "eligible_from": pd.to_datetime(["2022-05-12"]),
            "eligible_to": [pd.NaT],
            "exclusion_reason": ["test boundary"],
            "policy_type": ["economic_continuity"],
            "source_reference": ["test source"],
            "verification_status": ["test verified"],
            "decision_date": ["2026-10-28"],
        }
    )


def test_policy_boundaries_are_inclusive_and_security_specific() -> None:
    daily = pd.DataFrame(
        {
            "security_id": ["SEC:A", "SEC:A", "SEC:B"],
            "ticker": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2022-05-11", "2022-05-12", "2022-05-11"]),
            "volume": [1.0, 2.0, 3.0],
            "adj_close": [10.0, 11.0, 12.0],
            "pct_return": [float("nan")] * 3,
        }
    )
    retained, excluded = apply_eligibility_policy(daily, _policy())
    assert excluded["date"].tolist() == [pd.Timestamp("2022-05-11")]
    assert set(
        zip(retained["security_id"], retained["date"], strict=True)
    ) == {
        ("SEC:A", pd.Timestamp("2022-05-12")),
        ("SEC:B", pd.Timestamp("2022-05-11")),
    }
    assert set(retained["security_id"]) == {"SEC:A", "SEC:B"}
    assert retained.loc[retained["security_id"].eq("SEC:B"), "date"].item() == (
        pd.Timestamp("2022-05-11")
    )


def test_policy_rejects_unknown_security() -> None:
    daily = pd.DataFrame(
        {
            "security_id": ["SEC:B"],
            "date": pd.to_datetime(["2022-05-12"]),
        }
    )
    try:
        apply_eligibility_policy(daily, _policy())
    except ValueError as error:
        assert "absent securities" in str(error)
    else:
        raise AssertionError("Unknown policy security was not rejected")


def test_returns_do_not_cross_eligibility_boundary() -> None:
    calendar = xcals.get_calendar("XNYS")
    dates = calendar.sessions_in_range("2022-05-10", "2022-05-16")
    raw = pd.DataFrame(
        {
            "security_id": ["SEC:A"] * len(dates),
            "date": dates,
            "ticker": ["AAA"] * len(dates),
            "open": [10.0] * len(dates),
            "high": [11.0] * len(dates),
            "low": [9.0] * len(dates),
            "close": [10.0] * len(dates),
            "adj_close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100.0] * len(dates),
            "dividends": [0.0] * len(dates),
            "stock_splits": [0.0] * len(dates),
        }
    )
    result = recompute_diagnostics(
        raw,
        dates,
        return_threshold=0.5,
        volume_ratio_threshold=20.0,
        volume_lookback=20,
        volume_min_history=5,
    )
    # The retained segment begins at the first test row; no return is formed
    # from a previous observation outside the input segment.
    assert pd.isna(result.loc[0, "diagnostic_pct_return"])
    assert result.loc[1:, "diagnostic_pct_return"].notna().all()


def test_final_dataset_and_report_outputs() -> None:
    daily = pd.read_parquet(OUTPUT_ROOT / "selected_equity_daily.parquet")
    master = pd.read_parquet(OUTPUT_ROOT / "selected_security_master.parquet")
    exclusions = pd.read_csv(
        OUTPUT_ROOT / "historical_exclusions.csv", low_memory=False
    )
    zero = pd.read_csv(
        OUTPUT_ROOT / "remaining_zero_volume_cases.csv", low_memory=False
    )
    extremes = pd.read_csv(
        OUTPUT_ROOT / "remaining_extreme_value_cases.csv", low_memory=False
    )
    cleaning = json.loads(
        (OUTPUT_ROOT / "cleaning_policy.json").read_text(encoding="utf-8")
    )
    report_checks = json.loads(
        (
            PROJECT_ROOT
            / "reports/final_data_quality_report/generated/report_consistency_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert len(daily) == 1_186_196
    assert daily["security_id"].nunique() == 488
    assert master["cik"].nunique() == 486
    assert not daily.duplicated(["security_id", "date"]).any()
    assert daily["pct_return"].isna().all()
    assert exclusions.groupby("ticker").size().to_dict() == {
        "DKNG": 189,
        "FERG": 1601,
    }
    assert exclusions["original_source_file"].notna().all()
    assert set(exclusions["security_id"]) == {
        "SEC:0001832433",
        "SEC:0001883685",
    }
    for ticker, boundary in [("DKNG", "2020-04-24"), ("FERG", "2022-05-12")]:
        rows = exclusions[exclusions["ticker"].eq(ticker)]
        assert rows["date"].max() < boundary
        retained = daily.merge(
            master[["security_id", "ticker_2023"]],
            on="security_id",
        )
        assert retained.loc[
            retained["ticker_2023"].eq(ticker), "date"
        ].min() == pd.Timestamp(boundary)

    assert len(zero) == 16
    assert zero["zero_volume_classification"].eq(
        "vendor_confirmed_market_unverified"
    ).all()
    assert int(extremes["extreme_volume_spike_flag"].sum()) == 235
    assert int(extremes["extreme_price_flag"].sum()) == 0
    assert int(extremes["extreme_return_flag"].sum()) == 45
    assert cleaning["snapshot"]["fixed_membership_unchanged"]
    assert report_checks["passed"]
    assert report_checks["checks_passed"] == report_checks["checks_total"]


def test_exact_deliverables_exist() -> None:
    required = [
        "selected_equity_daily.parquet",
        "selected_security_master.parquet",
        "historical_exclusions.csv",
        "final_data_quality_summary.csv",
        "remaining_zero_volume_cases.csv",
        "remaining_extreme_value_cases.csv",
        "cleaning_policy.json",
    ]
    for name in required:
        assert (OUTPUT_ROOT / name).is_file(), name
    assert (PROJECT_ROOT / "reports/final_data_quality_report/main.pdf").is_file()
