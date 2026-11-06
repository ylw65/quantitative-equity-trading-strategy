from __future__ import annotations

from src.investigation.integrate_report import _tex


def test_latex_escape_handles_structured_values() -> None:
    assert _tex("ticker_&") == "ticker\\_\\allowbreak{}\\&"
    assert _tex(1234) == "1,234"
    assert _tex(2023) == "2023"
