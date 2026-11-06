from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common import PROJECT_ROOT
from src.investigation.security_identity_audit import cik10


AUDIT_ROOT = PROJECT_ROOT / "data/investigation/security_identity_audit_488"


def test_cik_format_preserves_leading_zeros_and_empty_values() -> None:
    assert cik10(9984) == "0000009984"
    assert cik10("756894") == "0000756894"
    assert cik10("") == ""
    assert cik10(None) == ""


def test_audit_covers_exactly_selected_488_securities() -> None:
    master = pd.read_parquet(
        PROJECT_ROOT
        / "data/filtered/adtv100m_2023_eligibility_v2/selected_security_master.parquet"
    )
    audit = pd.read_csv(
        AUDIT_ROOT / "security_identity_audit_488.csv",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    assert len(audit) == 488
    assert audit["security_id"].is_unique
    assert set(audit["security_id"]) == set(master["security_id"].astype(str))
    assert audit["cik"].str.len().eq(10).all()


def test_barnes_barrick_is_verified_wrong_issuer() -> None:
    audit = pd.read_csv(
        AUDIT_ROOT / "security_identity_audit_488.csv",
        dtype=str,
        keep_default_na=False,
    ).set_index("ticker_2023")
    row = audit.loc["B"]
    assert row["identity_audit_status"] == "vendor_series_mismatch"
    assert row["vendor_contamination_verified"] == "True"
    assert row["ticker_reuse_detected"] == "True"
    assert row["current_ticker_cik"] == "0000756894"

    reuse = pd.read_csv(
        AUDIT_ROOT / "ticker_reuse_candidates.csv",
        dtype=str,
        keep_default_na=False,
    )
    barnes = reuse[reuse["ticker"].eq("B")].iloc[0]
    assert barnes["original_cik"] == "0000009984"
    assert barnes["other_cik"] == "0000756894"
    assert barnes["original_last_verified_date"] == "2025-01-27"
    assert barnes["other_first_verified_date"] == "2025-05-09"
    assert barnes["contamination_status"] == "vendor_contamination_verified"

    events = pd.read_csv(
        AUDIT_ROOT / "historical_security_events.csv",
        dtype=str,
        keep_default_na=False,
    )
    b_events = events[events["security_id"].eq("SEC:0000009984")]
    assert set(b_events["event_type"]) == {
        "major_merger_and_delisting",
        "ticker_reused_by_different_issuer",
    }
    assert set(b_events["effective_date"]) == {"2025-01-27", "2025-05-09"}
    assert b_events["source_url"].ne("").all()


def test_ferg_and_dkng_are_distinct_event_types() -> None:
    audit = pd.read_csv(
        AUDIT_ROOT / "security_identity_audit_488.csv",
        dtype=str,
        keep_default_na=False,
    ).set_index("ticker_2023")
    assert audit.loc["FERG", "identity_audit_status"] == (
        "listing_eligibility_change"
    )
    assert audit.loc["DKNG", "identity_audit_status"] == "economic_regime_change"
    assert audit.loc["FERG", "vendor_contamination_verified"] == "False"
    assert audit.loc["DKNG", "vendor_contamination_verified"] == "False"

    events = pd.read_csv(
        AUDIT_ROOT / "historical_security_events.csv",
        dtype=str,
        keep_default_na=False,
    )
    ferg_id = audit.loc["FERG", "security_id"]
    dkng_id = audit.loc["DKNG", "security_id"]
    ferg_types = set(events.loc[events["security_id"].eq(ferg_id), "event_type"])
    dkng_types = set(events.loc[events["security_id"].eq(dkng_id), "event_type"])
    assert {"initial_nyse_listing", "primary_listing_change"}.issubset(ferg_types)
    assert "spac_business_combination" in dkng_types


def test_events_and_ids_reconcile_without_cross_issuer_merging() -> None:
    audit = pd.read_csv(
        AUDIT_ROOT / "security_identity_audit_488.csv",
        dtype=str,
        keep_default_na=False,
    )
    events = pd.read_csv(
        AUDIT_ROOT / "historical_security_events.csv",
        dtype=str,
        keep_default_na=False,
    )
    recon = pd.read_csv(
        AUDIT_ROOT / "vendor_series_reconciliation.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert events["event_id"].is_unique
    assert set(events["security_id"]).issubset(set(audit["security_id"]))
    assert set(recon["security_id"]).issubset(set(audit["security_id"]))
    event_counts = events.groupby("security_id").size().to_dict()
    for row in audit.itertuples(index=False):
        assert int(row.number_of_events) == event_counts.get(row.security_id, 0)


def test_summary_and_review_register_reconcile() -> None:
    audit = pd.read_csv(
        AUDIT_ROOT / "security_identity_audit_488.csv",
        dtype=str,
        keep_default_na=False,
    )
    review = pd.read_csv(
        AUDIT_ROOT / "identity_review_register.csv",
        dtype=str,
        keep_default_na=False,
    )
    summary = json.loads(
        (AUDIT_ROOT / "identity_audit_summary.json").read_text(encoding="utf-8")
    )
    assert summary["securities_audited"] == 488
    assert summary["no_conflict_detected"] == int(
        audit["identity_audit_status"].eq("no_conflict_detected").sum()
    )
    assert summary["potential_ticker_reuse"] == 1
    assert summary["verified_ticker_reuse"] == 1
    assert summary["verified_yahoo_attribution_errors"] == 1
    assert summary["unresolved_identity"] == 0
    assert summary["human_review"] == len(review) == 6
    assert review["review_status"].eq("pending_decision").all()
    assert summary["protected_inputs_unchanged"]


def test_required_audit_artifacts_exist() -> None:
    required = [
        "security_identity_audit_488.csv",
        "historical_security_events.csv",
        "ticker_reuse_candidates.csv",
        "vendor_series_reconciliation.csv",
        "identity_review_register.csv",
        "identity_audit_summary.md",
        "identity_audit_summary.json",
        "sec_filing_inventory.csv",
        "sec_submissions_cache_manifest.csv",
    ]
    for name in required:
        assert (AUDIT_ROOT / name).is_file(), name
    cache = pd.read_csv(AUDIT_ROOT / "sec_submissions_cache_manifest.csv")
    assert len(cache) == 491
    assert cache["status"].isin({"cached", "downloaded"}).all()
