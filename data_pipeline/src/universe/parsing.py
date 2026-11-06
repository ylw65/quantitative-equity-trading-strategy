"""Parsing and classification helpers for the historical universe."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable


PRIMARY_FORM_PATTERN = r"(?:10-K|10-Q|20-F|40-F)(?:/A)?"
INDEX_LINE = re.compile(
    rf"^(.*?)\s{{2,}}({PRIMARY_FORM_PATTERN})\s+(\d+)\s+"
    r"(\d{4}-\d{2}-\d{2})\s+(\S+)\s*$"
)


@dataclass(frozen=True)
class Filing:
    company_name: str
    form: str
    cik: str
    filed_date: date
    filename: str

    @property
    def accession(self) -> str:
        return self.filename.rsplit("/", 1)[-1].removesuffix(".txt")

    @property
    def url(self) -> str:
        return f"https://www.sec.gov/Archives/{self.filename}"


def parse_company_index(text: str) -> list[Filing]:
    """Parse SEC's fixed-width company.idx format without relying on widths."""
    filings: list[Filing] = []
    for line in text.splitlines():
        match = INDEX_LINE.match(line.rstrip())
        if not match:
            continue
        name, form, cik, filed, filename = match.groups()
        filings.append(
            Filing(
                company_name=" ".join(name.split()),
                form=form,
                cik=str(int(cik)),
                filed_date=date.fromisoformat(filed),
                filename=filename,
            )
        )
    return filings


def select_reference_filings(
    filings: Iterable[Filing], reference_date: date
) -> dict[str, Filing]:
    """Select the latest primary filing before reference, else earliest after."""
    grouped: dict[str, list[Filing]] = {}
    for filing in filings:
        grouped.setdefault(filing.cik, []).append(filing)

    selected: dict[str, Filing] = {}
    for cik, rows in grouped.items():
        before = [row for row in rows if row.filed_date <= reference_date]
        if before:
            selected[cik] = max(before, key=lambda row: row.filed_date)
        else:
            selected[cik] = min(rows, key=lambda row: row.filed_date)
    return selected


def _extract_tag(text: str, tag: str) -> list[tuple[str, str]]:
    patterns = (
        rf"<ix:[^>]+name=[\"']dei:{tag}[\"'][^>]*>(.*?)</ix:[^>]+>",
        rf"<dei:{tag}\b[^>]*>(.*?)</dei:{tag}>",
    )
    values: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            raw = match.group(1)
            context_match = re.search(
                r"contextRef=[\"']([^\"']+)", match.group(0), flags=re.IGNORECASE
            )
            context = context_match.group(1) if context_match else ""
            value = html.unescape(re.sub(r"<[^>]+>", " ", raw))
            value = " ".join(value.split())
            record = (context, value)
            if value and record not in values:
                values.append(record)
    return values


def extract_cover_securities(text: str) -> list[dict[str, str]]:
    """Extract securities listed under Section 12(b) from inline XBRL."""
    titles = _extract_tag(text, "Security12bTitle")
    tickers = _extract_tag(text, "TradingSymbol")
    exchanges = _extract_tag(text, "SecurityExchangeName")
    contexts: list[str] = []
    for records in (titles, tickers, exchanges):
        for context, _ in records:
            if context and context not in contexts:
                contexts.append(context)

    # Context IDs are the association key for multi-class or multi-instrument
    # cover tables. Sequence is not reliable: warrant and common facts can be
    # emitted in different orders.
    if contexts and all(contexts):
        by_context: dict[str, dict[str, str]] = {
            context: {"context_id": context} for context in contexts
        }
        for field, records in (
            ("security_title", titles),
            ("ticker", tickers),
            ("exchange", exchanges),
        ):
            for context, value in records:
                if context:
                    by_context[context][field] = value
        return [
            {
                "security_title": row.get("security_title", ""),
                "ticker": row.get("ticker", ""),
                "exchange": row.get("exchange", ""),
                "context_id": row["context_id"],
            }
            for row in by_context.values()
        ]

    # Fallback for non-inline XBRL fragments without contexts.
    title_values = [value for _, value in titles]
    ticker_values = [value for _, value in tickers]
    exchange_values = [value for _, value in exchanges]
    count = max(len(title_values), len(ticker_values), len(exchange_values))
    rows: list[dict[str, str]] = []
    for index in range(count):
        title = title_values[index] if index < len(title_values) else ""
        ticker = ticker_values[index] if index < len(ticker_values) else ""
        exchange = exchange_values[index] if index < len(exchange_values) else ""
        if ticker or title or exchange:
            rows.append(
                {
                    "security_title": title,
                    "ticker": ticker,
                    "exchange": exchange,
                    "context_id": "",
                }
            )
    return rows


def normalize_exchange(value: str) -> str:
    """Map exchange labels to the project's three eligible exchanges."""
    upper = " ".join(value.upper().replace(".", "").split())
    if "NASDAQ" in upper:
        return "Nasdaq"
    if (
        "NYSE AMERICAN" in upper
        or "NYSE MKT" in upper
        or "NYSEAMER" in upper
    ):
        return "NYSE American"
    if upper == "NYSE" or "NEW YORK STOCK EXCHANGE" in upper:
        return "NYSE"
    if "NYSE ARCA" in upper:
        return "NYSE Arca"
    if "CBOE" in upper or "CHICAGO BOARD OPTIONS" in upper:
        return "Cboe"
    return value.strip() or "Unknown"


def normalize_ticker(value: str) -> str:
    """Normalize SEC/Yahoo class separators while retaining the class suffix."""
    return (
        value.strip()
        .strip("\"'")
        .upper()
        .replace(".", "-")
        .replace("/", "-")
    )


def is_plausible_ticker(value: str) -> bool:
    """Reject SEC placeholders and malformed symbols before acquisition."""
    if value in {"", "N-A", "NONE", "NA", "0000000000"}:
        return False
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,4}", value))


def classify_security(title: str, company_name: str = "") -> tuple[str, str]:
    """Return (security_type, reason). An empty reason means eligible."""
    title_l = title.lower()
    company_l = company_name.lower()

    exclusions = (
        ("adr", ("american depositary", "depositary share", "depositary receipt", " adr")),
        ("preferred", ("preferred", "preference stock")),
        ("warrant", ("warrant",)),
        ("right", (" right", "rights to purchase", "rights to acquire")),
        ("unit", (" unit", "units representing")),
        ("debt", ("note", "debenture", "bond", "subordinated")),
        ("fund", ("exchange traded fund", " etf", "mutual fund", "closed-end fund")),
        (
            "trust",
            ("royalty trust", "income trust", "unit investment trust", "grantor trust"),
        ),
    )
    for security_type, needles in exclusions:
        if any(needle in title_l for needle in needles):
            return security_type, f"Title denotes {security_type}: {title}"

    common_markers = (
        "common stock",
        "common shares",
        "ordinary shares",
        "ordinary share",
        "class a common",
        "class b common",
        "class c common",
        "capital stock",
        "shares of beneficial interest",
        "common shares of beneficial interest",
    )
    if not any(marker in title_l for marker in common_markers):
        if not title:
            return "unknown", "No Section 12(b) security title was parsed"
        return "unknown", f"Security title is not clearly a common equity: {title}"

    if (
        "shares of beneficial interest" in title_l
        and any(word in company_l for word in ("reit", "realty", "real estate"))
    ):
        return "reit_common_stock", ""
    if any(word in title_l for word in ("class a", "class b", "class c")):
        return "common_equity_class", ""
    if "ordinary share" in title_l:
        return "ordinary_shares", ""
    return "common_stock", ""
