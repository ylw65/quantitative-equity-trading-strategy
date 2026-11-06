from __future__ import annotations

from pathlib import Path

import pytest

from src.filtering.liquidity_report_consistency import (
    _as_int,
    _macro_values,
    check_consistency,
)
from src.common import PROJECT_ROOT


def test_macro_parser_and_integer_format(tmp_path: Path) -> None:
    path = tmp_path / "macros.tex"
    path.write_text(
        "\\newcommand{\\SelectedSecurities}{488}\n"
        "\\newcommand{\\SelectedObservations}{1,187,986}\n"
    )
    values = _macro_values(path)
    assert _as_int(values["SelectedSecurities"]) == 488
    assert _as_int(values["SelectedObservations"]) == 1_187_986


def test_live_report_reconciles_with_filtered_snapshot() -> None:
    required = [
        PROJECT_ROOT / "data/filtered/adtv100m_2023/filter_metadata.json",
        PROJECT_ROOT
        / "data/filtered/adtv100m_2023/quality/post_filter_integrity_summary.json",
        PROJECT_ROOT / "reports/liquidity_report/generated/macros.tex",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("live liquidity report assets have not been generated")
    payload, checks = check_consistency(PROJECT_ROOT)
    assert payload["passed"], checks.loc[~checks["passed"]].to_string(index=False)
