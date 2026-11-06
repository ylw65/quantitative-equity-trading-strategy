from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.acquisition.download_eodhd_sample import (
    SampleRequest,
    normalize_responses,
    parse_price_csv,
    select_sample_requests,
)


PRICE = (
    b"Date,Open,High,Low,Close,Adjusted_close,Volume\n"
    b"2020-08-28,125.0,126.0,124.0,125.0,122.0,1000000\n"
    b"2020-08-31,127.0,128.0,126.0,127.0,124.0,4000000\n"
)
SPLITS = json.dumps(
    [{"date": "2020-08-31", "split": "4.000000/1.000000"}]
).encode()
DIVIDENDS = json.dumps(
    [
        {
            "date": "2020-08-28",
            "value": 0.205,
            "unadjustedValue": 0.82,
            "currency": "USD",
        }
    ]
).encode()


def request(ticker: str = "AAA") -> SampleRequest:
    return SampleRequest(
        security_id=f"SEC:{ticker}",
        ticker_2023=ticker,
        yahoo_ticker=ticker,
        eodhd_code=f"{ticker}.US",
        company_name=f"{ticker} Corp",
        exchange="NYSE",
        security_type="common_stock",
        current_status="current_sec_listing_match",
        selection_stratum="NYSE|common_stock|current_sec_listing_match",
    )


def test_price_and_actions_normalize_to_validation_schema() -> None:
    daily, n_splits, n_dividends = normalize_responses(
        request(), PRICE, SPLITS, DIVIDENDS, download_id="fixture"
    )
    assert n_splits == 1
    assert n_dividends == 1
    assert daily.loc[daily.date.eq(pd.Timestamp("2020-08-31")), "stock_splits"].item() == 4
    assert daily.loc[daily.date.eq(pd.Timestamp("2020-08-28")), "dividends"].item() == 0.82
    assert daily["source"].eq("eodhd_public_demo").all()
    assert "pct_return" not in daily.columns


def test_price_csv_errors_are_not_filled() -> None:
    frame = parse_price_csv(PRICE)
    assert frame["volume"].tolist() == [1000000, 4000000]
    try:
        parse_price_csv(b"Forbidden")
    except ValueError as exc:
        assert "unexpected price response" in str(exc)
    else:
        raise AssertionError("invalid price body unexpectedly parsed")


def test_sample_selection_is_unique_and_stratified(tmp_path: Path) -> None:
    rows = []
    for exchange, prefix in [("NYSE", "Y"), ("Nasdaq", "Q")]:
        for kind_number, kind in enumerate(
            ["common_stock", "common_equity_class"]
        ):
            for index in range(4):
                ticker = f"{prefix}{kind_number}{index}"
                rows.append(
                    {
                        "security_id": f"SEC:{ticker}",
                        "ticker_2023": ticker,
                        "yahoo_ticker": ticker,
                        "company_name": f"{ticker} Corp",
                        "exchange": exchange,
                        "security_type": kind,
                        "current_ticker": ticker if index % 2 == 0 else "",
                    }
                )
    path = tmp_path / "master.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    selected = select_sample_requests(
        path,
        limit=8,
        seed=1,
        explicit_tickers=["Q00"],
    )
    assert len(selected) == 8
    assert len({row.yahoo_ticker for row in selected}) == 8
    assert selected[0].yahoo_ticker == "Q00"
    assert len({row.selection_stratum for row in selected}) == 8
