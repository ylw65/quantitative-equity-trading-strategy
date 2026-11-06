from __future__ import annotations

import pandas as pd

from src.investigation.missing_analysis import (
    _group_comparison,
    _normalise_inputs,
    _recovery_candidates,
    classify_failure,
    parse_change_targets,
)


def test_failure_classification_does_not_equate_provider_miss_with_delisting() -> None:
    assert (
        classify_failure(
            "YFPricesMissingError('possibly delisted; no price data found')"
        )
        == "provider_symbol_unavailable"
    )
    assert (
        classify_failure("YFRateLimitError('Too Many Requests')")
        == "provider_rate_limit"
    )
    assert classify_failure("SSLError: connection closed") == "provider_transport_error"


def test_parse_change_targets_handles_single_and_multi_symbols() -> None:
    assert parse_change_targets("OLD -> NEW") == ["NEW"]
    assert parse_change_targets("OLD -> NEW,NEWW") == ["NEW", "NEWW"]
    assert parse_change_targets("") == []


def test_missing_cohort_rates_are_computed_from_manifest() -> None:
    master = pd.DataFrame(
        {
            "security_id": ["SEC:1", "SEC:2"],
            "ticker_2023": ["AAA", "BBB"],
            "yahoo_ticker": ["AAA", "BBB"],
            "company_name": ["A", "B"],
            "exchange": ["Nasdaq", "Nasdaq"],
            "security_type": ["common_stock"] * 2,
            "current_ticker": ["", ""],
            "subsequent_ticker_changes": ["AAA -> CCC", ""],
            "listing_date": ["", ""],
            "delisting_date": ["", ""],
            "source_confidence": ["high", "high"],
        }
    )
    manifest = pd.DataFrame(
        {
            "security_id": ["SEC:1", "SEC:2"],
            "download_status": ["failed", "success"],
            "error": [
                "YFTzMissingError('possibly delisted; no timezone found')",
                "",
            ],
        }
    )
    result = _normalise_inputs(master, manifest)
    assert result["is_available"].tolist() == [False, True]
    assert result.loc[result.security_id.eq("SEC:1"), "evidence_category"].iloc[0] == (
        "ticker_change_alias_unresolved"
    )
    group = _group_comparison(result, "exchange")
    assert group.loc[0, "unavailable"] == 1
    assert group.loc[0, "acquisition_rate"] == 0.5


def test_preferred_like_later_symbol_is_not_a_safe_common_equity_alias() -> None:
    failed = pd.DataFrame(
        [
            {
                "security_id": "SEC:1:AEL",
                "ticker_2023": "AEL",
                "company_name": "Example",
                "exchange": "NYSE",
                "security_type": "common_stock",
                "subsequent_ticker_changes": "AEL -> ANG-PD",
            }
        ]
    )
    result = _recovery_candidates(failed)
    assert not bool(result.loc[0, "safe_single_alias"])
    assert result.loc[0, "candidate_status"] == (
        "preferred_or_noncommon_symbol_suspect"
    )
