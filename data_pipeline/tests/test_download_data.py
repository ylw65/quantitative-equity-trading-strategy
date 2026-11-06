from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.acquisition.download_data import (
    REQUIRED_VENDOR_FIELDS,
    SecurityRequest,
    _is_permanent_unavailable_error,
    _is_permanent_request_unavailable,
    acquire,
    load_requests,
    normalize_vendor_frame,
)


def sample_config(tmp_path: Path, retries: int = 1) -> dict:
    return {
        "project": {
            "start_date": "2016-01-01",
            "end_date_exclusive": "2026-01-01",
        },
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "metadata_dir": str(tmp_path / "metadata"),
        },
        "download": {
            "batch_size": 2,
            "retries": retries,
            "backoff_seconds": 0,
            "request_pause_seconds": 0,
            "timeout_seconds": 5,
        },
        "research_constraints": {"calculate_final_pct_return": False},
    }


def sample_frame(start: str = "2016-01-04") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=3)
    return pd.DataFrame(
        {
            "Open": [10.0, 10.5, 11.0],
            "High": [10.8, 11.0, 11.4],
            "Low": [9.8, 10.3, 10.8],
            "Close": [10.5, 10.9, 11.2],
            "Adj Close": [9.9, 10.3, 10.6],
            "Volume": [1000, 1100, 1200],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=pd.Index(dates, name="Date"),
    )


def test_explicit_tickers_are_smoke_only(tmp_path: Path) -> None:
    requests = load_requests(tmp_path / "missing.parquet", explicit_tickers=["AAPL"])
    assert requests[0].security_id == "TICKER2023:AAPL"
    assert requests[0].yahoo_ticker == "AAPL"


def test_normalize_preserves_required_and_extra_fields() -> None:
    request = SecurityRequest("SEC:1:AAPL", "AAPL", "AAPL")
    frame = sample_frame()
    frame["Capital Gains"] = 0.0
    result, missing = normalize_vendor_frame(
        frame, request, "2016-01-01", "2026-01-01", "test-run"
    )
    assert missing == []
    assert REQUIRED_VENDOR_FIELDS.issubset(result.columns)
    assert "capital_gains" in result.columns
    assert result["date"].is_monotonic_increasing
    assert result["ticker"].eq("AAPL").all()


def test_all_nan_vendor_placeholder_is_not_an_observation() -> None:
    request = SecurityRequest("SEC:1:DELISTED", "DELISTED", "DELISTED")
    frame = sample_frame()
    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        frame[column] = float("nan")
    result, _ = normalize_vendor_frame(
        frame, request, "2016-01-01", "2026-01-01", "test-run"
    )
    assert result.empty


def test_permanent_unavailable_errors_are_distinguished_from_retryable() -> None:
    assert _is_permanent_unavailable_error(
        "YFTzMissingError('possibly delisted; no timezone found')"
    )
    assert _is_permanent_unavailable_error(
        "YFPricesMissingError('possibly delisted; no price data found')"
    )
    assert not _is_permanent_unavailable_error(
        "YFRateLimitError('Too Many Requests. Rate limited.')"
    )
    assert not _is_permanent_unavailable_error("SSLError: connection closed")


def test_current_ticker_is_not_permanent_after_transport_failure() -> None:
    delisted = SecurityRequest("SEC:1:GONE", "GONE", "GONE")
    current = SecurityRequest(
        "SEC:2:LIVE", "LIVE", "LIVE", current_ticker="LIVE"
    )
    missing = "YFTzMissingError('possibly delisted; no timezone found')"
    assert _is_permanent_request_unavailable(delisted, missing)
    assert not _is_permanent_request_unavailable(current, missing)


def test_batch_failure_isolated_and_partial_recorded(tmp_path: Path) -> None:
    frame = sample_frame()
    calls: list[list[str]] = []

    def fake_downloader(tickers, **kwargs):
        calls.append(tickers)
        if tickers == ["BAD"]:
            raise RuntimeError("simulated vendor failure")
        ticker_frames = {}
        for ticker in tickers:
            ticker_frames[ticker] = frame
        data = pd.concat(ticker_frames, axis=1)
        data.columns.names = ["Ticker", "Price"]
        return data

    requests = [
        SecurityRequest("SEC:1:AAA", "AAA", "AAA"),
        SecurityRequest("SEC:2:BBB", "BBB", "BBB"),
        SecurityRequest("SEC:3:BAD", "BAD", "BAD"),
    ]
    manifest, failures = acquire(
        requests,
        sample_config(tmp_path),
        raw_dir=tmp_path / "raw",
        manifest_path=tmp_path / "metadata" / "manifest.csv",
        failures_path=tmp_path / "metadata" / "failures.csv",
        resume=False,
        downloader=fake_downloader,
    )
    assert set(manifest.loc[manifest.download_status == "success", "yahoo_ticker"]) == {
        "AAA",
        "BBB",
    }
    assert failures.loc[failures.yahoo_ticker == "BAD", "download_status"].eq(
        "failed"
    ).all()
    assert all(len(tickers) <= 2 for tickers in calls)


def test_resume_keeps_existing_raw_checkpoint(tmp_path: Path) -> None:
    config = sample_config(tmp_path)
    request = SecurityRequest("SEC:1:AAPL", "AAPL", "AAPL")
    frame = sample_frame()

    def first_downloader(tickers, **kwargs):
        return frame

    acquire(
        [request],
        config,
        raw_dir=tmp_path / "raw",
        manifest_path=tmp_path / "metadata" / "manifest.csv",
        failures_path=tmp_path / "metadata" / "failures.csv",
        resume=False,
        downloader=first_downloader,
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("resume unexpectedly redownloaded a checkpoint")

    manifest, _ = acquire(
        [request],
        config,
        raw_dir=tmp_path / "raw",
        manifest_path=tmp_path / "metadata" / "manifest.csv",
        failures_path=tmp_path / "metadata" / "failures.csv",
        resume=True,
        downloader=should_not_run,
    )
    assert manifest.loc[0, "download_status"] == "success"
