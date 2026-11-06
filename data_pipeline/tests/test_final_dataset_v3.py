from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common import PROJECT_ROOT


V2 = PROJECT_ROOT / "data/filtered/adtv100m_2023_eligibility_v2"
V3 = PROJECT_ROOT / "data/filtered/adtv100m_2023_eligibility_v3"
B_ID = "SEC:0000009984"


def test_v3_membership_and_row_counts() -> None:
    daily = pd.read_parquet(V3 / "selected_equity_daily.parquet")
    master = pd.read_parquet(V3 / "selected_security_master.parquet")
    diagnostics = pd.read_parquet(V3 / "diagnostic_returns.parquet")
    assert len(master) == 487
    assert master["cik"].nunique() == 485
    assert len(daily) == 1_183_682
    assert len(diagnostics) == 1_186_192
    assert daily["security_id"].nunique() == 487
    assert not daily.duplicated(["security_id", "date"]).any()
    assert not diagnostics.duplicated(["security_id", "date"]).any()
    assert daily["pct_return"].isna().all()


def test_b_security_is_removed_only_from_v3() -> None:
    v2_daily = pd.read_parquet(V2 / "selected_equity_daily.parquet")
    v3_daily = pd.read_parquet(V3 / "selected_equity_daily.parquet")
    v3_master = pd.read_parquet(V3 / "selected_security_master.parquet")
    v3_diag = pd.read_parquet(V3 / "diagnostic_returns.parquet")
    assert int(v2_daily["security_id"].astype(str).eq(B_ID).sum()) == 2_514
    assert not v3_daily["security_id"].astype(str).eq(B_ID).any()
    assert not v3_master["security_id"].astype(str).eq(B_ID).any()
    assert not v3_diag["security_id"].astype(str).eq(B_ID).any()

    removed = pd.read_csv(
        V3 / "removed_security_observations.csv", low_memory=False
    )
    assert len(removed) == 2_514
    assert removed["security_id"].eq(B_ID).all()
    assert removed["ticker"].eq("B").all()


def test_security_exclusion_audit_is_complete() -> None:
    exclusions = pd.read_csv(V3 / "security_exclusions.csv", dtype=str)
    assert len(exclusions) == 1
    row = exclusions.iloc[0]
    assert row["security_id"] == B_ID
    assert row["cik"] == "0000009984"
    assert row["ticker"] == "B"
    assert row["observations_removed"] == "2514"
    assert row["first_observation"] == "2016-01-04"
    assert row["last_observation"] == "2025-12-31"
    assert row["policy_type"] == "verified_identity_contamination"
    assert row["original_source_file"] == "data/raw/yahoo/B__416316778b.parquet"


def test_v2_historical_cutoffs_and_exclusions_are_preserved() -> None:
    daily = pd.read_parquet(V3 / "selected_equity_daily.parquet")
    master = pd.read_parquet(V3 / "selected_security_master.parquet")
    historical = pd.read_csv(V3 / "historical_exclusions.csv", low_memory=False)
    assert historical.groupby("ticker").size().to_dict() == {
        "DKNG": 189,
        "FERG": 1601,
    }
    joined = daily.merge(
        master[["security_id", "ticker_2023"]],
        on="security_id",
        how="left",
    )
    assert joined.loc[joined.ticker_2023.eq("DKNG"), "date"].min() == (
        pd.Timestamp("2020-04-24")
    )
    assert joined.loc[joined.ticker_2023.eq("FERG"), "date"].min() == (
        pd.Timestamp("2022-05-12")
    )


def test_quality_counts_are_recomputed_for_v3() -> None:
    diagnostics = pd.read_parquet(V3 / "diagnostic_returns.parquet")
    summary = pd.read_csv(V3 / "final_data_quality_summary.csv")
    assert int(diagnostics["zero_volume_flag"].sum()) == 16
    assert int(diagnostics["extreme_volume_spike_flag"].sum()) == 235
    assert int(diagnostics["extreme_return_flag"].sum()) == 45
    assert int(diagnostics["zero_or_negative_adj_close_flag"].sum()) == 0
    metrics = summary.set_index("metric")["final_research_dataset"]
    assert metrics["Securities"] == 487
    assert metrics["Distinct SEC issuers"] == 485
    assert metrics["Validated stock-day observations"] == 1_183_682
    assert metrics["B security-removal observations"] == 2_514
    assert metrics["Zero-volume observations"] == 16
    assert metrics["Extreme-volume observations"] == 235
    assert metrics["Extreme diagnostic-return observations"] == 45


def test_v3_metadata_and_read_only_guarantee() -> None:
    execution = json.loads(
        (V3 / "execution_summary.json").read_text(encoding="utf-8")
    )
    policy = json.loads(
        (V3 / "cleaning_policy.json").read_text(encoding="utf-8")
    )
    assert execution["snapshot"]["securities"] == 487
    assert execution["snapshot"]["issuers"] == 485
    assert execution["snapshot"]["security_identity_exclusions"] == 1
    assert execution["snapshot"]["security_identity_exclusion_rows"] == 2_514
    assert execution["preserved_historical_exclusion_rows"] == 1_790
    assert execution["protected_inputs_unchanged"]
    assert policy["parent_version"] == "adtv100m_2023_eligibility_v2"
    assert policy["security_exclusion_policy"][0]["ticker"] == "B"


def test_required_v3_artifacts_exist() -> None:
    required = [
        "selected_equity_daily.parquet",
        "selected_security_master.parquet",
        "diagnostic_returns.parquet",
        "historical_exclusions.csv",
        "security_exclusions.csv",
        "removed_security_observations.csv",
        "final_data_quality_summary.csv",
        "remaining_zero_volume_cases.csv",
        "remaining_zero_volume_by_security.csv",
        "remaining_extreme_value_cases.csv",
        "security_coverage.csv",
        "cleaning_policy.json",
        "execution_summary.json",
        "README.md",
    ]
    for name in required:
        assert (V3 / name).is_file(), name
