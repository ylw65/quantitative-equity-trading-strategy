from __future__ import annotations

import pandas as pd

from src.acquisition.download_data import (
    SecurityRequest,
    normalize_vendor_frame,
)
from src.investigation.identity_temporal import _fingerprint


def test_all_nan_placeholder_is_not_a_price_observation() -> None:
    dates = pd.bdate_range("2023-01-03", periods=3)
    frame = pd.DataFrame(
        {
            "Open": [None, None, None],
            "High": [None, None, None],
            "Low": [None, None, None],
            "Close": [None, None, None],
            "Adj Close": [None, None, None],
            "Volume": [None, None, None],
            "Dividends": [None, None, None],
            "Stock Splits": [None, None, None],
        },
        index=pd.Index(dates, name="Date"),
    )
    result, _ = normalize_vendor_frame(
        frame,
        SecurityRequest("SEC:1:NONE", "NONE", "NONE"),
        "2016-01-01",
        "2026-01-01",
        "test",
    )
    assert result.empty


def test_exact_fingerprint_detects_identical_active_history() -> None:
    dates = pd.bdate_range("2023-01-03", periods=25)
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": range(100, 125),
            "adj_close": range(100, 125),
            "volume": range(1000, 1025),
        }
    )
    same = frame.copy()
    other = frame.assign(volume=range(2000, 2025))
    assert _fingerprint(frame)[0] == _fingerprint(same)[0]
    assert _fingerprint(frame)[0] != _fingerprint(other)[0]
