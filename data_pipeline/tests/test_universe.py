from __future__ import annotations

from datetime import date

from src.universe.parsing import (
    classify_security,
    extract_cover_securities,
    is_plausible_ticker,
    normalize_exchange,
    parse_company_index,
    select_reference_filings,
)


def test_parse_and_select_reference_filings() -> None:
    text = """Description: test

Company Name                                                  Form Type   CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
Example Corp                                                  10-Q             12345      2022-11-01  edgar/data/12345/0000000001-22-000001.txt
Example Corp                                                  10-K             12345      2023-02-10  edgar/data/12345/0000000002-23-000001.txt
"""
    filings = parse_company_index(text)
    selected = select_reference_filings(filings, date(2023, 1, 3))
    assert selected["12345"].form == "10-Q"
    assert selected["12345"].filed_date == date(2022, 11, 1)


def test_extract_multiclass_cover() -> None:
    text = """
    <ix:nonNumeric name="dei:Security12bTitle">Class A Common Stock</ix:nonNumeric>
    <ix:nonNumeric name="dei:TradingSymbol">ABC.A</ix:nonNumeric>
    <ix:nonNumeric name="dei:SecurityExchangeName">The New York Stock Exchange</ix:nonNumeric>
    <ix:nonNumeric name="dei:Security12bTitle">Class B Common Stock</ix:nonNumeric>
    <ix:nonNumeric name="dei:TradingSymbol">ABC.B</ix:nonNumeric>
    <ix:nonNumeric name="dei:SecurityExchangeName">NYSE</ix:nonNumeric>
    """
    rows = extract_cover_securities(text)
    assert [(row["ticker"], row["exchange"]) for row in rows] == [
        ("ABC.A", "The New York Stock Exchange"),
        ("ABC.B", "NYSE"),
    ]


def test_security_classification() -> None:
    assert classify_security("Class A Common Stock")[0] == "common_equity_class"
    assert classify_security("Ordinary Shares")[0] == "ordinary_shares"
    assert classify_security("American Depositary Shares")[0] == "adr"
    assert classify_security("7.25% Preferred Stock")[0] == "preferred"
    assert classify_security("Warrants to purchase common stock")[0] == "warrant"
    assert classify_security("Rights to acquire common shares")[0] == "right"


def test_exchange_normalization() -> None:
    assert normalize_exchange("The Nasdaq Global Select Market") == "Nasdaq"
    assert normalize_exchange("NYSE American") == "NYSE American"
    assert normalize_exchange("New York Stock Exchange") == "NYSE"


def test_ticker_placeholders() -> None:
    assert is_plausible_ticker("BRK-B")
    assert not is_plausible_ticker("N-A")
    assert not is_plausible_ticker("EXC-28")
