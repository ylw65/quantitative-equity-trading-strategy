from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.validate_data import (
    ISSUE_COLUMNS,
    IssueCollector,
    identity_audit,
    standardize_frame,
    validate,
)


def fixture_config(tmp_path: Path) -> dict:
    return {
        "project": {
            "start_date": "2016-01-01",
            "end_date_exclusive": "2026-01-01",
        },
        "paths": {"metadata_dir": str(tmp_path / "metadata")},
        "validation": {
            "us_calendar": "XNYS",
            "minimum_valid_sessions": 252,
            "extreme_return_abs_threshold": 0.50,
            "extreme_volume_ratio_threshold": 20.0,
            "max_unexpected_gap_sessions": 10,
            "split_relative_tolerance": 0.03,
            "dividend_adjustment_tolerance": 0.01,
        },
    }


def vendor_frame(security_id: str, ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-05"]
            ),
            "ticker": ticker,
            "security_id": security_id,
            "ticker_2023": ticker,
            "provider": "yahoo_finance",
            "download_id": "fixture",
            "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
            "open": [100.0, 101.0, 102.0, 102.0],
            "high": [102.0, 103.0, 104.0, 104.0],
            "low": [99.0, 100.0, 101.0, 101.0],
            "close": [101.0, 102.0, 103.0, 103.0],
            "adj_close": [101.0, 102.0, 103.0, 103.0],
            "volume": [1000.0, 1100.0, 1200.0, 1200.0],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_standardize_keeps_raw_duplicates_but_cleans_base() -> None:
    result = standardize_frame(vendor_frame("SEC:0000000001", "AAA"))
    assert len(result.audit) == 4
    assert len(result.clean) == 3
    assert result.duplicate_rows_removed == 1
    assert result.diagnostics["diagnostic_pct_return"].notna().sum() == 2


def test_split_continuity_valid_event_not_flagged() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "ticker": ["AAA", "AAA"],
            "security_id": ["SEC:0000000001"] * 2,
            "provider": ["eodhd_public_demo", "eodhd_public_demo"],
            "open": [50.0, 25.0],
            "high": [51.0, 26.0],
            "low": [49.0, 24.0],
            "close": [50.0, 25.0],
            "adj_close": [25.0, 25.0],
            "volume": [1000.0, 2000.0],
            "dividends": [0.0, 0.0],
            "stock_splits": [0.0, 2.0],
        }
    )
    collector = IssueCollector()
    config = fixture_config(Path("/tmp"))
    runtime = type(
        "Runtime",
        (),
        {"config": config, "issues": collector},
    )()
    from src.validation.validate_data import _corporate_action_checks

    split_flags, dividend_flags = _corporate_action_checks(
        standardize_frame(frame).clean,
        "AAA",
        "SEC:0000000001",
        0.03,
        0.01,
        collector,
    )
    assert split_flags == 0
    assert dividend_flags == 0


def test_split_continuity_handles_nonzero_market_move() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-08-28", "2020-08-31"]),
            "ticker": ["AAPL", "AAPL"],
            "security_id": ["SEC:0000320193:AAPL"] * 2,
            "provider": ["eodhd_public_demo", "eodhd_public_demo"],
            "open": [500.0, 127.0],
            "high": [510.0, 130.0],
            "low": [495.0, 124.0],
            "close": [499.23, 129.04],
            # Raw return is about -35.4%; adjusted gross return is about 3.39%.
            "adj_close": [117.77, 121.76],
            "volume": [46_000_000.0, 225_000_000.0],
            "dividends": [0.0, 0.0],
            "stock_splits": [0.0, 4.0],
        }
    )
    collector = IssueCollector()
    from src.validation.validate_data import _corporate_action_checks

    split_flags, _ = _corporate_action_checks(
        standardize_frame(frame).clean,
        "AAPL",
        "SEC:0000320193:AAPL",
        0.03,
        0.01,
        collector,
    )
    assert split_flags == 0
    assert "split_adjustment_continuity" not in set(
        collector.frame()["issue_type"]
    )


def test_yahoo_split_close_is_already_split_continuous() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-07-12", "2022-07-13"]),
            "ticker": ["YHOO", "YHOO"],
            "security_id": ["SEC:0000000003:YHOO"] * 2,
            "provider": ["yahoo_finance", "yahoo_finance"],
            "open": [25.0, 25.0],
            "high": [26.0, 26.0],
            "low": [24.0, 24.0],
            "close": [25.0, 25.0],
            # Yahoo nominal Close is back-adjusted; both series are continuous
            # across a 2:1 event and should not imply a 2x factor jump.
            "adj_close": [25.0, 25.0],
            "volume": [1000.0, 2100.0],
            "dividends": [0.0, 0.0],
            "stock_splits": [0.0, 2.0],
        }
    )
    collector = IssueCollector()
    from src.validation.validate_data import _corporate_action_checks

    split_flags, _ = _corporate_action_checks(
        standardize_frame(frame).clean,
        "YHOO",
        "SEC:0000000003:YHOO",
        0.03,
        0.01,
        collector,
    )
    assert split_flags == 0


def test_combined_split_and_distribution_is_not_simple_continuity_failure() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-03", "2023-05-04"]),
            "ticker": ["COMB", "COMB"],
            "security_id": ["SEC:0000000004:COMB"] * 2,
            "provider": ["yahoo_finance", "yahoo_finance"],
            "open": [100.0, 40.0],
            "high": [102.0, 43.0],
            "low": [99.0, 39.0],
            "close": [100.0, 42.0],
            "adj_close": [100.0, 42.0],
            "volume": [1000.0, 2000.0],
            "dividends": [0.0, 8.0],
            "stock_splits": [0.0, 2.0],
        }
    )
    collector = IssueCollector()
    from src.validation.validate_data import _corporate_action_checks

    split_flags, dividend_flags = _corporate_action_checks(
        standardize_frame(frame).clean,
        "COMB",
        "SEC:0000000004:COMB",
        0.03,
        0.01,
        collector,
    )
    assert split_flags == 0
    assert dividend_flags == 0
    assert "simultaneous_split_and_distribution" in set(
        collector.frame()["issue_type"]
    )


def test_validate_outputs_null_final_returns_and_issue_register(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = vendor_frame("SEC:0000000001", "AAA")
    raw = raw.loc[raw["date"].ne(pd.Timestamp("2023-01-04"))]
    raw.to_parquet(raw_dir / "AAA.parquet", index=False)

    master = pd.DataFrame(
        {
            "security_id": ["SEC:0000000001"],
            "ticker_2023": ["AAA"],
            "yahoo_ticker": ["AAA"],
            "company_name": ["Fixture A"],
            "exchange": ["Nasdaq"],
            "security_type": ["common_stock"],
            "listing_date": ["2023-01-03"],
            "delisting_date": ["2023-01-05"],
            "included": [True],
        }
    )
    master_path = tmp_path / "master.parquet"
    master.to_parquet(master_path, index=False)
    result = validate(
        fixture_config(tmp_path),
        raw_dir,
        tmp_path / "validated",
        master_path,
        metadata_dir=tmp_path / "metadata",
        data_mode="fixture",
    )
    assert result["base_rows"] == 2
    base = pd.read_parquet(tmp_path / "validated" / "base.parquet")
    assert base["pct_return"].isna().all()
    assert not base.duplicated(["security_id", "date"]).any()

    diagnostics = pd.read_parquet(
        tmp_path / "validated" / "diagnostic_returns.parquet"
    )
    assert "diagnostic_pct_return" in diagnostics.columns
    issues = pd.read_csv(tmp_path / "validated" / "data_issues.csv")
    assert set(issues.columns) == set(ISSUE_COLUMNS)
    assert "duplicate_security_date_rows" in set(issues["issue_type"])
    assert "unobserved_sessions_inside_history" in set(issues["issue_type"])
    quality = pd.read_csv(
        tmp_path / "validated" / "data_quality_by_security.csv"
    )
    assert quality.loc[0, "verified_active_missing_sessions"] == 1
    assert bool(quality.loc[0, "fewer_than_252_valid_sessions"]) is True


def test_zero_and_negative_values_are_flagged_not_replaced() -> None:
    frame = vendor_frame("SEC:0000000002", "BBB").iloc[:2].copy()
    frame["adj_close"] = [0.0, np.nan]
    frame["volume"] = [-1.0, 0.0]
    result = standardize_frame(frame)
    diagnostics = result.diagnostics
    assert diagnostics["zero_or_negative_adj_close_flag"].iloc[0]
    assert diagnostics["negative_volume_flag"].iloc[0]
    assert diagnostics["zero_volume_flag"].iloc[1]
    assert result.clean["adj_close"].isna().iloc[1]


def test_identity_audit_detects_ticker_reuse() -> None:
    master = pd.DataFrame(
        {
            "security_id": ["SEC:0000000001", "SEC:0000000002"],
            "yahoo_ticker": ["SAME", "SAME"],
            "ticker_2023": ["SAME", "SAME"],
            "reference_date_verified": [True, True],
        }
    )
    issues = IssueCollector()
    findings = identity_audit(master, issues)
    assert "ticker_mapped_to_multiple_security_ids" in set(findings["finding"])
    register = issues.frame()
    assert "ticker_reuse_or_ambiguous_mapping" in set(register["issue_type"])


def test_raw_adjusted_conflict_is_flagged_without_action() -> None:
    frame = vendor_frame("SEC:0000000003", "CCC").iloc[:2].copy()
    frame["close"] = [100.0, 105.0]
    frame["adj_close"] = [100.0, 100.0]
    result = standardize_frame(frame)
    assert bool(result.diagnostics["raw_adjusted_return_conflict_flag"].iloc[1])
    assert result.clean["adj_close"].notna().all()
