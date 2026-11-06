from __future__ import annotations

from src.investigation.report_consistency import LEGACY_ABSTRACT_PHRASES


def test_legacy_failure_narrative_is_detected() -> None:
    stale = "The primary Yahoo Finance request path was unavailable."
    assert any(phrase in stale for phrase in LEGACY_ABSTRACT_PHRASES)
