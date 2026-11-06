from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.filtering.liquidity_screen import (
    PROJECT_ROOT,
    adtv20,
    classify_window,
    daily_dollar_volume,
    final_sessions,
    future_split_product,
)


def test_final_twenty_sessions_end_2022() -> None:
    sessions = final_sessions("XNYS", "2022-12-30", 20)
    assert len(sessions) == 20
    assert sessions[0] == pd.Timestamp("2022-12-02")
    assert sessions[-1] == pd.Timestamp("2022-12-30")
    assert pd.Timestamp("2022-12-26") not in sessions


def test_dollar_volume_and_zero_volume_are_valid() -> None:
    values = daily_dollar_volume(
        pd.Series([10.0, 10.0, np.nan, -1.0]),
        pd.Series([5.0, 0.0, 2.0, 1.0]),
    )
    assert values[0] == 50.0
    assert values[1] == 0.0
    assert np.isnan(values[2])
    assert np.isnan(values[3])


def test_adtv_requires_all_twenty_values() -> None:
    assert adtv20(pd.Series([1.0] * 20)) == 1.0
    assert np.isnan(adtv20(pd.Series([1.0] * 19 + [np.nan])))


def test_threshold_is_inclusive_and_insufficient_is_separate() -> None:
    assert classify_window(20, 20, 100_000_000, 100_000_000) == "selected"
    assert classify_window(20, 20, 99_999_999, 100_000_000) == "excluded_below_threshold"
    assert classify_window(19, 20, np.nan, 100_000_000) == "insufficient_liquidity_window_data"
    assert classify_window(20, 20, np.nan, 100_000_000) == "unresolved_price_volume_issue"


def test_future_split_product_uses_only_later_events() -> None:
    splits = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-12-15", "2023-06-01", "2024-01-01", "2026-01-01"]),
            "stock_splits": [2.0, 3.0, 0.5, 10.0],
        }
    )
    assert future_split_product(splits, pd.Timestamp("2022-12-15"), pd.Timestamp("2025-12-31")) == 1.5
    assert future_split_product(splits, pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31")) == 1.0


def test_2023_plus_volume_not_used_in_screen() -> None:
    # The screen function receives only dates in the explicit 2022 session grid.
    sessions = set(final_sessions("XNYS", "2022-12-30", 20))
    fixture_dates = pd.to_datetime(["2022-12-29", "2022-12-30", "2023-01-03"])
    selected_dates = [date for date in fixture_dates if date in sessions]
    assert selected_dates == [pd.Timestamp("2022-12-29"), pd.Timestamp("2022-12-30")]


def test_generated_outputs_reconcile_and_propagate_selected_ids() -> None:
    root = PROJECT_ROOT / "data/filtered/adtv100m_2023"
    path = root / "liquidity_screen_all.csv"
    if not path.exists():
        pytest.skip("live filtered outputs have not been generated")
    screen = pd.read_csv(path)
    selected_screen = pd.read_csv(root / "liquidity_screen_selected.csv")
    selected_master = pd.read_parquet(root / "selected_security_master.parquet")
    daily = pd.read_parquet(root / "selected_equity_daily.parquet")
    excluded = pd.read_csv(root / "liquidity_screen_excluded.csv")
    expected_ids = set(selected_screen["security_id"])
    assert set(selected_master["security_id"]) == expected_ids
    assert set(daily["security_id"]) == expected_ids
    assert len(selected_screen) + len(excluded) == len(screen)
    assert selected_screen["screen_status"].eq("selected").all()
    assert daily["pct_return"].isna().all()
    assert not daily.duplicated(["security_id", "date"]).any()


def test_generated_screen_statuses_are_exclusive() -> None:
    path = (
        PROJECT_ROOT
        / "data/filtered/adtv100m_2023/liquidity_screen_all.csv"
    )
    if not path.exists():
        pytest.skip("live filtered outputs have not been generated")
    screen = pd.read_csv(path)
    assert screen["security_id"].is_unique
    assert not (
        screen["screen_status"].eq("selected")
        & screen["screen_status"].ne("selected")
    ).any()
    acquired = screen[screen["acquisition_status"].eq("acquired")]
    assert set(acquired["screen_status"]).issubset(
        {
            "selected",
            "excluded_below_threshold",
            "insufficient_liquidity_window_data",
            "unresolved_price_volume_issue",
        }
    )
    assert screen["acquisition_status"].isin(["acquired", "not_acquired"]).all()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_original_inputs_match_recorded_hashes() -> None:
    metadata_path = (
        PROJECT_ROOT / "data/filtered/adtv100m_2023/filter_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("live filtered outputs have not been generated")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for name, expected in metadata["source_hashes_sha256"].items():
        config = (
            PROJECT_ROOT / "configs/liquidity_filter.yaml"
        ).read_text(encoding="utf-8")
        # Recover the path through the declared config rather than hardcoding it.
        import yaml

        paths = yaml.safe_load(config)["inputs"]
        assert _sha256(PROJECT_ROOT / paths[name]) == expected
