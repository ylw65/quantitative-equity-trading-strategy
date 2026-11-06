from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import PROJECT_ROOT
from src.filtering.ohlc_enrichment import (
    OUTPUT_COLUMNS,
    _validate_unique,
    left_join_ohlc,
)


V3 = PROJECT_ROOT / "data/filtered/adtv100m_2023_eligibility_v3"
OHLC = PROJECT_ROOT / "data/filtered/adtv100m_2023_eligibility_v3_ohlc"


def test_duplicate_join_keys_fail_explicitly() -> None:
    frame = pd.DataFrame(
        {
            "security_id": ["SEC:A", "SEC:A"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        }
    )
    with pytest.raises(RuntimeError, match="duplicate security-date keys"):
        _validate_unique(frame, "test source")


def test_missing_ohlc_source_row_preserves_null_values() -> None:
    daily = pd.DataFrame(
        {
            "security_id": ["SEC:A"],
            "date": pd.to_datetime(["2020-01-02"]),
            "ticker": ["AAA"],
            "volume": [100.0],
            "adj_close": [10.0],
            "pct_return": [np.nan],
        }
    )
    source = pd.DataFrame(
        columns=[
            "security_id",
            "date",
            "source_file",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
        ]
    )
    joined = left_join_ohlc(daily, source)
    assert len(joined) == 1
    assert not joined.loc[0, "ohlc_source_row_present"]
    assert pd.isna(joined.loc[0, "source_open"])
    assert joined.loc[0, "adj_close"] == 10.0
    assert joined.loc[0, "volume"] == 100.0


def test_v3_security_date_grid_is_exactly_preserved() -> None:
    v3 = pd.read_parquet(V3 / "selected_equity_daily.parquet")
    enriched = pd.read_parquet(OHLC / "selected_equity_daily.parquet")
    v3_keys = set(
        zip(
            v3["security_id"].astype(str),
            pd.to_datetime(v3["date"]),
            strict=True,
        )
    )
    out_keys = set(
        zip(
            enriched["security_id"].astype(str),
            pd.to_datetime(enriched["date"]),
            strict=True,
        )
    )
    assert len(enriched) == 1_181_168
    assert out_keys == {
        key for key in v3_keys if key[0] != "SEC:0000048898"
    }
    assert not enriched.duplicated(["security_id", "date"]).any()


def test_existing_v3_fields_are_unchanged() -> None:
    v3 = pd.read_parquet(V3 / "selected_equity_daily.parquet")
    v3 = v3[
        ~v3["security_id"].astype(str).eq("SEC:0000048898")
    ].reset_index(drop=True)
    enriched = pd.read_parquet(OHLC / "selected_equity_daily.parquet").reset_index(
        drop=True
    )
    assert list(enriched.columns) == OUTPUT_COLUMNS
    assert v3["security_id"].equals(enriched["security_id"])
    assert v3["date"].equals(enriched["date"])
    assert v3["ticker"].equals(enriched["ticker"])
    for column in ("volume", "adj_close"):
        assert np.allclose(
            v3[column].to_numpy(dtype=float),
            enriched[column].to_numpy(dtype=float),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-10,
        )
    assert enriched["pct_return"].isna().all()


def test_v3_master_and_exclusion_boundaries_are_preserved() -> None:
    v3_master = pd.read_parquet(V3 / "selected_security_master.parquet")
    master = pd.read_parquet(OHLC / "selected_security_master.parquet")
    daily = pd.read_parquet(OHLC / "selected_equity_daily.parquet")
    expected_master = v3_master[
        ~v3_master["security_id"].astype(str).eq("SEC:0000048898")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(master.reset_index(drop=True), expected_master)
    assert len(master) == 486
    assert master["cik"].nunique() == 484
    assert not daily["security_id"].astype(str).eq("SEC:0000048898").any()
    assert not daily["security_id"].astype(str).eq("SEC:0000009984").any()
    joined = daily.merge(
        master[["security_id", "ticker_2023"]],
        on="security_id",
        how="left",
    )
    assert joined.loc[joined.ticker_2023.eq("FERG"), "date"].min() == (
        pd.Timestamp("2022-05-12")
    )
    assert joined.loc[joined.ticker_2023.eq("DKNG"), "date"].min() == (
        pd.Timestamp("2020-04-24")
    )


def test_ohlc_completeness_and_consistency_register() -> None:
    daily = pd.read_parquet(OHLC / "selected_equity_daily.parquet")
    issues = pd.read_csv(OHLC / "ohlc_discrepancies.csv", low_memory=False)
    metadata = json.loads(
        (OHLC / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    assert daily[["open", "high", "low", "close"]].notna().all().all()
    assert len(daily) == 1_181_168
    assert metadata["complete_ohlc_rows"] == len(daily)
    assert metadata["complete_ohlc_pct"] == 100.0
    assert metadata["securities_with_complete_ohlc"] == 486
    assert metadata["rows_missing_any_ohlc"] == 0
    assert metadata["source_conflict_records"] == 0
    assert metadata["records_requiring_targeted_download"] == 0
    assert metadata["inconsistent_ohlc_rows"] == 0
    assert metadata["ohlc_issue_records"] == 0
    assert metadata["security_exclusions"] == 1
    assert metadata["security_exclusion_observations"] == 2_514
    assert metadata["vendor_repaired_rows"] == 3_532
    assert metadata["vendor_repaired_securities"] == 5
    assert issues.empty


def test_source_provenance_and_schema_comparison() -> None:
    provenance = json.loads(
        (OHLC / "ohlc_source_provenance.json").read_text(encoding="utf-8")
    )
    schema = pd.read_csv(OHLC / "schema_comparison.csv")
    coverage = pd.read_csv(OHLC / "ohlc_coverage_report.csv")
    metadata = json.loads((OHLC / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert provenance["primary_source"] == "data/validated/standardized_raw.parquet"
    assert provenance["identity_join_keys"] == ["security_id", "date"]
    assert provenance["targeted_yahoo_requests"] == 0
    assert provenance["sources_modified"] is False
    assert len(coverage) == 486
    assert coverage["complete_ohlc_pct"].eq(100.0).all()
    added = set(schema.loc[schema["change"].eq("added"), "field"])
    assert {
        "open",
        "high",
        "low",
        "close",
        "dividends",
        "stock_splits",
        "vendor_repair_flag",
    } <= added
    assert metadata["parent_dataset"] == "adtv100m_2023_eligibility_v3"
    assert metadata["v3_securities"] == 487
    assert metadata["protected_inputs_unchanged"]


def test_required_artifacts_exist() -> None:
    required = [
        "selected_equity_daily.parquet",
        "selected_security_master.parquet",
        "ohlc_coverage_report.csv",
        "ohlc_discrepancies.csv",
        "ohlc_source_provenance.json",
        "dataset_metadata.json",
        "security_exclusions.csv",
        "removed_security_observations.csv",
        "schema_comparison.csv",
        "OHLC_ENRICHMENT_REPORT.md",
    ]
    for name in required:
        assert (OHLC / name).is_file(), name


def test_hubb_security_exclusion_audit() -> None:
    exclusions = pd.read_csv(OHLC / "security_exclusions.csv", dtype=str)
    removed = pd.read_csv(
        OHLC / "removed_security_observations.csv", low_memory=False
    )
    assert len(exclusions) == 1
    row = exclusions.iloc[0]
    assert row["security_id"] == "SEC:0000048898"
    assert row["ticker"] == "HUBB"
    assert row["observations_removed"] == "2514"
    assert row["first_observation"] == "2016-01-04"
    assert row["last_observation"] == "2025-12-31"
    assert row["policy_type"] == "user_directed_security_exclusion"
    assert len(removed) == 2_514
    assert removed["security_id"].eq("SEC:0000048898").all()
    assert removed["ticker"].eq("HUBB").all()
    assert removed[["open", "high", "low", "close"]].notna().all().all()
