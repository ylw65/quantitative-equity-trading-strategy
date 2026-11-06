#!/usr/bin/env python3
"""Generate and validate a clearly labelled synthetic integration fixture.

This script exists only to exercise the pipeline when live Yahoo acquisition is
unavailable. Its outputs must never be presented as empirical market data.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DEFAULT_CONFIG, load_config, project_path
from src.validation.validate_data import load_calendar_sessions, validate


def make_security(
    index: int,
    sessions: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    ticker = f"FIX{index:03d}"
    security_id = f"FIXTURE:SEC{index:03d}"
    end = sessions[-1]
    listing_date = sessions[0]
    delisting_date: pd.Timestamp | pd.NaT = pd.NaT
    security_dates = sessions
    if index == 8:
        security_dates = sessions[sessions <= pd.Timestamp("2024-08-30")]
        end = security_dates[-1]
        delisting_date = end
    if index == 9:
        listing_date = pd.Timestamp("2019-01-02")
        security_dates = sessions[sessions >= listing_date]

    count = len(security_dates)
    gross = 1.0 + rng.normal(0.0003, 0.012 if index < 4 else 0.018, count)
    total_index = np.cumprod(gross)
    stock_splits = np.zeros(count)
    if index == 2:
        split_position = int(np.searchsorted(security_dates, pd.Timestamp("2022-06-01")))
        stock_splits[split_position] = 2.0
    split_events = np.where(stock_splits > 0, stock_splits, 1.0)
    cumulative_split = np.cumprod(split_events)
    final_split = cumulative_split[-1]
    close = 80.0 * total_index / cumulative_split
    open_ = close * (1.0 + rng.normal(0.0, 0.004, count))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.015, count))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.015, count))
    volume = rng.lognormal(13.0, 0.35, count) * cumulative_split
    dividends = np.zeros(count)

    if index == 3:
        quarter_dates = {"2021-03-15", "2021-06-15", "2021-09-15", "2021-12-15"}
        dividend_mask = security_dates.isin(pd.to_datetime(sorted(quarter_dates)))
        dividends[dividend_mask] = close[dividend_mask] * 0.005
    if index == 4:
        volume[100:105] = 0
    if index == 5:
        volume[300] = -1
    if index == 7:
        close[400] *= 2.2

    current_share_price = close * cumulative_split / final_split
    previous_current_price = np.r_[current_share_price[0], current_share_price[:-1]]
    economic_return = (
        current_share_price + dividends
    ) / previous_current_price - 1.0
    economic_return[0] = 0.0
    wealth = np.cumprod(1.0 + economic_return)
    adj_close = current_share_price[-1] * wealth / wealth[-1]

    data = pd.DataFrame(
        {
            "date": security_dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj_close,
            "volume": volume,
            "dividends": dividends,
            "stock_splits": stock_splits,
        }
    )
    if index == 1:
        data = data.drop(index=data.index[500:503])
    if index == 0:
        data = pd.concat([data, data.iloc[[700]]], ignore_index=True)
    data = data.sort_values("date", kind="stable")
    data.insert(1, "ticker", ticker)
    data.insert(2, "security_id", security_id)
    data.insert(3, "ticker_2023", ticker)
    data.insert(4, "provider", "synthetic_fixture")
    data.insert(5, "download_id", "fixture-v1")
    data.insert(6, "retrieved_at_utc", "2026-10-26T00:00:00+00:00")

    exchanges = ["Nasdaq", "NYSE", "NYSE American"]
    security_types = ["common_stock", "common_equity_class", "reit_common_stock"]
    metadata = {
        "security_id": security_id,
        "cik": "",
        "ticker_2023": ticker,
        "yahoo_ticker": ticker,
        "company_name": f"Fixture Security {index}",
        "exchange": exchanges[index % len(exchanges)],
        "security_type": security_types[index % len(security_types)],
        "permanent_id_type": "synthetic_fixture",
        "permanent_id": security_id,
        "listing_date": listing_date,
        "delisting_date": delisting_date,
        "subsequent_ticker_changes": "",
        "included": True,
        "exclusion_reason": "",
        "source": "pipeline_fixture_not_market_data",
        "source_confidence": "synthetic",
        "reference_date_verified": True,
    }
    return data, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output-dir", default="data/validation_fixture"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_path(args.output_dir)
    raw_dir = root / "raw"
    metadata_dir = root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_calendar_sessions("XNYS", "2016-01-01", "2025-12-31")
    rng = np.random.default_rng(5205)
    master_rows = []
    manifest_rows = []
    for index in range(10):
        frame, metadata = make_security(index, sessions, rng)
        path = raw_dir / f"{metadata['ticker_2023']}.parquet"
        frame.to_parquet(path, index=False)
        master_rows.append(metadata)
        manifest_rows.append(
            {
                "security_id": metadata["security_id"],
                "ticker_2023": metadata["ticker_2023"],
                "yahoo_ticker": metadata["yahoo_ticker"],
                "company_name": metadata["company_name"],
                "exchange": metadata["exchange"],
                "download_status": "success",
                "history_status": "fixture",
                "n_observations": len(frame),
                "raw_path": str(path.relative_to(project_path("."))),
                "error": "",
                "fixture_only": True,
            }
        )
    master = pd.DataFrame(master_rows)
    master.to_csv(root / "security_master.csv", index=False)
    master.to_parquet(root / "security_master.parquet", index=False)
    pd.DataFrame(manifest_rows).to_csv(
        metadata_dir / "download_manifest.csv", index=False
    )
    pd.DataFrame(columns=["security_id", "yahoo_ticker", "download_status", "error"]).to_csv(
        metadata_dir / "download_failures.csv", index=False
    )
    result = validate(
        config,
        raw_dir=raw_dir,
        output_dir=root / "validated",
        master_path=root / "security_master.parquet",
        metadata_dir=metadata_dir,
        data_mode="fixture",
    )
    (root / "FIXTURE_ONLY.md").write_text(
        "\n".join(
            [
                "# Synthetic pipeline fixture",
                "",
                "Every observation in this directory is synthetic and was created",
                "only to test the acquisition/validation/reporting interfaces.",
                "Do not interpret these values as US equity market evidence.",
                "",
                f"Validated base rows: {result['base_rows']}",
                f"Securities: {result['securities_with_raw_data']}",
                f"Issues: {result['issues']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(
        (root / "FIXTURE_ONLY.md").read_bytes()
    ).hexdigest()
    print(f"Fixture validation complete: {result}")
    print(f"FIXTURE_ONLY.md SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
