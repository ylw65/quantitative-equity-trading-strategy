"""Reconstruct the best verifiable 3 January 2023 US common-equity universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from src.common import (
    PROJECT_ROOT,
    atomic_write_json,
    configure_logging,
    load_config,
    project_path,
)
from src.universe.parsing import (
    Filing,
    classify_security,
    extract_cover_securities,
    is_plausible_ticker,
    normalize_exchange,
    normalize_ticker,
    parse_company_index,
    select_reference_filings,
)
from src.universe.sec_client import SecClient


LOGGER = logging.getLogger(__name__)
METADATA_COLUMNS = [
    "security_id",
    "ticker_2023",
    "yahoo_ticker",
    "company_name",
    "exchange",
    "security_type",
    "permanent_id_type",
    "permanent_id",
    "listing_date",
    "delisting_date",
    "subsequent_ticker_changes",
    "included",
    "exclusion_reason",
    "source",
    "source_confidence",
    "reference_date_verified",
    "reference_date_verification_method",
    "cik",
    "filing_form",
    "filing_date",
    "filing_url",
    "security_title",
    "current_ticker",
    "current_exchange",
    "notes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_if_needed(
    client: SecClient, url: str, path: Path, force: bool = False
) -> Path:
    if path.exists() and not force:
        return path
    LOGGER.info("Downloading %s", url)
    response = client.get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    return path


def _load_index(path: Path) -> list[Filing]:
    return parse_company_index(path.read_text(encoding="utf-8", errors="replace"))


def _load_current_mapping(path: Path) -> dict[str, list[dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = payload["fields"]
    rows = payload["data"]
    mapping: dict[str, list[dict[str, str]]] = defaultdict(list)
    for values in rows:
        record = dict(zip(fields, values))
        mapping[str(int(record["cik"]))].append(
            {
                "ticker": normalize_ticker(str(record.get("ticker") or "")),
                "exchange": str(record.get("exchange") or ""),
                "name": str(record.get("name") or ""),
            }
        )
    return mapping


def _cache_path(cache_dir: Path, filing: Filing) -> Path:
    digest = hashlib.sha1(filing.url.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"CIK{filing.cik}_{digest}.json"


def _extract_one(
    client: SecClient,
    filing: Filing,
    cache_dir: Path,
    probe_bytes: int,
    force: bool,
) -> dict[str, Any]:
    path = _cache_path(cache_dir, filing)
    if path.exists() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        securities = cached.get("securities", [])
        # Version the association method: old multi-security caches were
        # sequence-zipped and must be refreshed rather than silently reused.
        if len(securities) <= 1 or all(
            "context_id" in security for security in securities
        ):
            return cached

    result: dict[str, Any] = {
        "cik": filing.cik,
        "filing_url": filing.url,
        "filing_form": filing.form,
        "filing_date": filing.filed_date.isoformat(),
        "company_name": filing.company_name,
        "status": "ok",
        "error": "",
        "securities": [],
        "bytes_requested": probe_bytes,
    }
    try:
        text = client.get_filing_probe(filing.url, probe_bytes)
        securities = extract_cover_securities(text)
        result["securities"] = securities
        result["probe_text_characters"] = len(text)
        if not securities:
            result["status"] = "no_cover_tags"
            result["error"] = (
                "No Section 12(b) cover tags found in the bounded filing probe"
            )
    except Exception as exc:  # noqa: BLE001 - failure is recorded per filing
        result["status"] = "request_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    atomic_write_json(result, path)
    return result


def _safe_confidence(
    filing: Filing,
    security_type: str,
    exchange: str,
    allowed_exchanges: set[str],
    reference_date: str,
) -> tuple[bool, str, str]:
    eligible_type = bool(security_type) and security_type not in {
        "unknown",
        "adr",
        "preferred",
        "warrant",
        "right",
        "unit",
        "debt",
        "fund",
        "trust",
    }
    eligible_exchange = exchange in allowed_exchanges
    if not eligible_type or not eligible_exchange:
        return False, "low", "classification_failed_or_out_of_scope"
    if filing.filed_date.isoformat() <= reference_date:
        return (
            True,
            "high",
            "latest_primary_filing_on_or_before_reference_date",
        )
    return (
        False,
        "medium",
        "first_primary_filing_after_reference_date",
    )


def _issuer_quality(name: str) -> int:
    """Score likely parent issuers when one cover symbol repeats across CIKs."""
    lower = name.lower()
    score = 0
    for marker, weight in (
        ("inc", 5),
        (" corp", 5),
        ("corporation", 5),
        ("trust", 4),
        ("plc", 4),
        ("holdings", 3),
        (" group", 3),
        (" l.p.", -7),
        (" lp", -7),
        (" llc", -7),
        ("operating partnership", -8),
        (" finance", -4),
        (" gas co", -3),
        (" electric co", -3),
    ):
        if marker in lower:
            score += weight
    return score


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    cik: str,
    ticker: str,
    issue_type: str,
    severity: str,
    field: str,
    message: str,
    source: str,
) -> None:
    key = f"{cik}|{ticker}|{issue_type}|{field}|{message}"
    issues.append(
        {
            "issue_id": "UNI-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12],
            "severity": severity,
            "issue_type": issue_type,
            "cik": cik,
            "ticker": ticker,
            "field": field,
            "message": message,
            "source": source,
            "resolution_status": "open",
        }
    )


def _build_from_results(
    selected: dict[str, Filing],
    results: list[dict[str, Any]],
    current_mapping: dict[str, list[dict[str, str]]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    reference_date = str(config["project"]["reference_date"])
    allowed = set(config["universe"]["allowed_exchanges"])
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    parsed_by_cik = {row["cik"]: row for row in results}

    for cik, filing in selected.items():
        result = parsed_by_cik[cik]
        current = current_mapping.get(cik, [])
        if result["status"] != "ok":
            current_listed = [
                item for item in current if normalize_exchange(item["exchange"]) in allowed
            ]
            if current_listed:
                for item in current_listed:
                    rows.append(
                        {
                            "security_id": f"SEC:{int(cik):010d}:{item['ticker']}",
                            "ticker_2023": item["ticker"],
                            "yahoo_ticker": item["ticker"],
                            "company_name": filing.company_name,
                            "exchange": normalize_exchange(item["exchange"]),
                            "security_type": "unknown",
                            "permanent_id_type": "SEC_CIK",
                            "permanent_id": str(int(cik)),
                            "listing_date": "",
                            "delisting_date": "",
                            "subsequent_ticker_changes": "",
                            "included": False,
                            "exclusion_reason": (
                                "Current listing exists but common-equity status and "
                                "reference-date listing were not verified from the filing probe"
                            ),
                            "source": "SEC filing index + current SEC ticker reference",
                            "source_confidence": "low",
                            "reference_date_verified": False,
                            "reference_date_verification_method": "unverified_supplement",
                            "cik": cik,
                            "filing_form": filing.form,
                            "filing_date": filing.filed_date.isoformat(),
                            "filing_url": filing.url,
                            "security_title": "",
                            "current_ticker": item["ticker"],
                            "current_exchange": item["exchange"],
                            "notes": "Excluded from verified master pending evidence",
                        }
                    )
            _add_issue(
                issues,
                cik=cik,
                ticker=current[0]["ticker"] if current else "",
                issue_type=result["status"],
                severity="high" if current_listed else "medium",
                field="security_master",
                message=result["error"],
                source=filing.url,
            )
            continue

        securities = result["securities"]
        if not securities:
            _add_issue(
                issues,
                cik=cik,
                ticker="",
                issue_type="no_cover_tags",
                severity="medium",
                field="ticker_2023",
                message="Filing parsed but no Section 12(b) cover table was found",
                source=filing.url,
            )
            continue

        seen_for_cik: set[str] = set()
        for position, security in enumerate(securities):
            ticker = normalize_ticker(security["ticker"])
            exchange = normalize_exchange(security["exchange"])
            security_type, exclusion_reason = classify_security(
                security["security_title"], filing.company_name
            )
            if not ticker:
                _add_issue(
                    issues,
                    cik=cik,
                    ticker=ticker,
                    issue_type="missing_ticker",
                    severity="high",
                    field="ticker_2023",
                    message=f"Cover security has no trading symbol: {security}",
                    source=filing.url,
                )
                continue
            plausible_ticker = is_plausible_ticker(ticker)
            if not plausible_ticker:
                _add_issue(
                    issues,
                    cik=cik,
                    ticker=ticker,
                    issue_type="invalid_ticker",
                    severity="high",
                    field="ticker_2023",
                    message=f"Cover security has a missing/malformed symbol: {security}",
                    source=filing.url,
                )
            if ticker in seen_for_cik:
                _add_issue(
                    issues,
                    cik=cik,
                    ticker=ticker,
                    issue_type="duplicate_cover_security",
                    severity="medium",
                    field="ticker_2023",
                    message="Duplicate cover security tags within one filing",
                    source=filing.url,
                )
                continue
            seen_for_cik.add(ticker)

            exact_current = next(
                (item for item in current if item["ticker"] == ticker), None
            )
            reconciled_exchange = False
            if exchange not in allowed and exact_current is not None:
                mapped_exchange = normalize_exchange(exact_current["exchange"])
                if mapped_exchange in allowed:
                    exchange = mapped_exchange
                    reconciled_exchange = True

            if (
                "acquisition corp" in filing.company_name.lower()
                and ticker.endswith(("U", "W", "R"))
                and security_type in {"common_stock", "common_equity_class"}
            ):
                security_type = (
                    "unit"
                    if ticker.endswith("U")
                    else "warrant"
                    if ticker.endswith("W")
                    else "right"
                )
                exclusion_reason = (
                    f"Acquisition-company symbol suffix denotes {security_type}: {ticker}"
                )

            verified, confidence, method = _safe_confidence(
                filing, security_type, exchange, allowed, reference_date
            )
            if not plausible_ticker:
                verified = False
                confidence = "low"
                method = "invalid_or_placeholder_ticker"
                exclusion_reason = (
                    f"Ticker is not a plausible exchange symbol: {ticker}"
                )
            if verified and reconciled_exchange:
                confidence = "medium"
                method = (
                    "pre_reference_cover_title_with_exact_ticker_"
                    "current_exchange_reconciliation"
                )
            included = verified and not exclusion_reason
            if not exclusion_reason and not verified:
                exclusion_reason = (
                    "Eligible type/exchange, but listing on the reference date is "
                    "not established by the selected SEC filing"
                )
            if not exclusion_reason and exchange not in allowed:
                exclusion_reason = f"Exchange is outside scope: {exchange}"

            security_id = f"SEC:{int(cik):010d}"
            if position > 0 or len(securities) > 1:
                security_id += f":{ticker}"

            current_tickers = [item["ticker"] for item in current if item["ticker"]]
            current_ticker = ticker if ticker in current_tickers else ""
            current_exchange = next(
                (
                    item["exchange"]
                    for item in current
                    if item["ticker"] == ticker
                ),
                "",
            )
            changes = ""
            if current_tickers and ticker not in current_tickers:
                changes = f"{ticker} -> {','.join(current_tickers)}"
            row = {
                "security_id": security_id,
                "ticker_2023": ticker,
                "yahoo_ticker": ticker,
                "company_name": filing.company_name,
                "exchange": exchange,
                "security_type": security_type,
                "permanent_id_type": "SEC_CIK",
                "permanent_id": str(int(cik)),
                "listing_date": "",
                "delisting_date": "",
                "subsequent_ticker_changes": changes,
                "included": included,
                "exclusion_reason": exclusion_reason,
                "source": "SEC primary filing cover (inline XBRL)",
                "source_confidence": confidence,
                "reference_date_verified": verified,
                "reference_date_verification_method": method,
                "cik": cik,
                "filing_form": filing.form,
                "filing_date": filing.filed_date.isoformat(),
                "filing_url": filing.url,
                "security_title": security["security_title"],
                "current_ticker": current_ticker,
                "current_exchange": current_exchange,
                "notes": (
                    "Presence on the reference date is inferred from SEC listing and "
                    "filings; it must be confirmed by reference-date market data."
                ),
            }
            rows.append(row)

            if security_type == "unknown":
                _add_issue(
                    issues,
                    cik=cik,
                    ticker=ticker,
                    issue_type="ambiguous_security_type",
                    severity="medium",
                    field="security_type",
                    message=exclusion_reason,
                    source=filing.url,
                )
            if not verified and not exclusion_reason:
                _add_issue(
                    issues,
                    cik=cik,
                    ticker=ticker,
                    issue_type="reference_date_not_verified",
                    severity="high",
                    field="reference_date_verified",
                    message=method,
                    source=filing.url,
                )
            if changes:
                _add_issue(
                    issues,
                    cik=cik,
                    ticker=ticker,
                    issue_type="subsequent_ticker_change",
                    severity="info",
                    field="subsequent_ticker_changes",
                    message=(
                        f"2023 ticker differs from current SEC ticker: {changes}; "
                        "intermediate history is unknown"
                    ),
                    source=filing.url,
                )

    universe = pd.DataFrame(rows)
    if not universe.empty:
        # Parent companies and operating subsidiaries sometimes tag the same
        # listed symbol. Retain both audit rows but only one security in scope.
        included_mask = universe["included"].astype(bool)
        for ticker, group in universe[included_mask].groupby("ticker_2023"):
            if len(group) < 2:
                continue

            def rank(row: dict[str, Any]) -> tuple[int, int, str]:
                exact_current = int(row["current_ticker"] == ticker)
                return (
                    exact_current * 100
                    + _issuer_quality(str(row["company_name"]))
                    + int(row["reference_date_verified"]) * 10,
                    int(str(row["filing_date"]).replace("-", "")),
                    str(row["security_id"]),
                )

            ordered = sorted(group.to_dict("records"), key=rank, reverse=True)
            winner = ordered[0]
            for row in ordered[1:]:
                mask = universe["security_id"].eq(row["security_id"]) & universe[
                    "cik"
                ].eq(row["cik"])
                universe.loc[mask, "included"] = False
                universe.loc[mask, "exclusion_reason"] = (
                    "Duplicate listed ticker resolved to canonical issuer "
                    f"{winner['company_name']} ({winner['cik']})"
                )
                _add_issue(
                    issues,
                    cik=row["cik"],
                    ticker=ticker,
                    issue_type="duplicate_security_resolved",
                    severity="medium",
                    field="included",
                    message=(
                        "Excluded duplicate issuer representation for one listed "
                        f"ticker; retained CIK {winner['cik']}"
                    ),
                    source=row["filing_url"],
                )

        duplicates = universe[
            universe["ticker_2023"].duplicated(keep=False)
            & universe["ticker_2023"].ne("")
        ]
        for ticker, group in duplicates.groupby("ticker_2023"):
            ciks = sorted(group["cik"].unique().tolist())
            for cik in ciks:
                _add_issue(
                    issues,
                    cik=cik,
                    ticker=ticker,
                    issue_type="ticker_reuse_or_collision",
                    severity="high",
                    field="ticker_2023",
                    message=f"Ticker occurs for multiple CIKs: {ciks}",
                    source="cross-security audit",
                )
    return universe, issues


def run_build(
    config_path: str | Path,
    *,
    force: bool = False,
    workers: int = 6,
) -> dict[str, int | str]:
    """Execute the full SEC-based universe build with resumable caches."""
    config = load_config(config_path)
    metadata_dir = project_path(config["paths"]["metadata_dir"])
    universe_dir = project_path(config["paths"]["universe_dir"])
    cache_dir = metadata_dir / "sec_cover_cache"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    universe_dir.mkdir(parents=True, exist_ok=True)
    log = configure_logging(project_path(config["paths"]["logs_dir"]) / "universe.log")
    global LOGGER
    LOGGER = log

    universe_cfg = config["universe"]
    client = SecClient(
        requests_per_second=float(universe_cfg["sec_requests_per_second"]),
        timeout_seconds=float(universe_cfg["request_timeout_seconds"]),
        retries=int(universe_cfg["retries"]),
    )
    q4_path = _download_if_needed(
        client,
        universe_cfg["sec_filing_index_2022_q4_url"],
        metadata_dir / "sec_company_idx_2022_q4.idx",
        force=force,
    )
    q1_path = _download_if_needed(
        client,
        universe_cfg["sec_filing_index_url"],
        metadata_dir / "sec_company_idx_2023_q1.idx",
        force=force,
    )
    tickers_path = _download_if_needed(
        client,
        universe_cfg["sec_current_tickers_url"],
        metadata_dir / "sec_company_tickers_exchange.json",
        force=force,
    )

    q4 = _load_index(q4_path)
    q1 = _load_index(q1_path)
    selected = select_reference_filings(
        q4 + q1, date.fromisoformat(str(config["project"]["reference_date"]))
    )
    LOGGER.info(
        "Selected %d CIKs from %d Q4-2022 and %d Q1-2023 primary filings",
        len(selected),
        len(q4),
        len(q1),
    )

    probe_bytes = int(universe_cfg["sec_compressed_probe_bytes"])
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _extract_one, client, filing, cache_dir, probe_bytes, force
            ): filing.cik
            for filing in selected.values()
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                counts = Counter(row["status"] for row in results)
                LOGGER.info(
                    "Parsed %d/%d filings; statuses=%s",
                    completed,
                    len(futures),
                    dict(counts),
                )

    # A primary filing can omit Section 12(b) tags or have an incomplete bounded
    # response. Retry only those CIKs against another Q1 filing when available.
    q1_by_cik: dict[str, Filing] = {}
    for filing in q1:
        existing = q1_by_cik.get(filing.cik)
        if existing is None or filing.filed_date < existing.filed_date:
            q1_by_cik[filing.cik] = filing
    fallback: dict[str, Filing] = {}
    for cik, result in {row["cik"]: row for row in results}.items():
        candidate = q1_by_cik.get(cik)
        if (
            result["status"] != "ok"
            and candidate is not None
            and candidate.url != selected[cik].url
        ):
            fallback[cik] = candidate

    fallback_results: list[dict[str, Any]] = []
    if fallback:
        LOGGER.info("Retrying %d failed/blank covers against alternate filings", len(fallback))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    _extract_one, client, filing, cache_dir, probe_bytes, force
                ): filing.cik
                for filing in fallback.values()
            }
            for count, future in enumerate(as_completed(futures), start=1):
                fallback_results.append(future.result())
                if count == 1 or count % 100 == 0 or count == len(futures):
                    LOGGER.info(
                        "Retry parsed %d/%d filings", count, len(futures)
                    )

    selected_for_build = dict(selected)
    fallback_by_cik = {row["cik"]: row for row in fallback_results}
    for cik, result in fallback_by_cik.items():
        selected_for_build[cik] = fallback[cik]
    results.extend(fallback_results)

    current_mapping = _load_current_mapping(tickers_path)
    universe, issues = _build_from_results(
        selected_for_build, results, current_mapping, config
    )
    universe = universe[METADATA_COLUMNS]
    included = universe[universe["included"]].copy()
    included = included.sort_values(["ticker_2023", "security_id"]).reset_index(drop=True)
    universe = universe.sort_values(
        ["included", "ticker_2023", "security_id"], ascending=[False, True, True]
    ).reset_index(drop=True)

    universe_path = universe_dir / "universe_2023.csv"
    master_path = universe_dir / "security_master.parquet"
    issues_path = universe_dir / "universe_issues.csv"
    universe.to_csv(universe_path, index=False)
    included.to_parquet(master_path, index=False)
    issue_frame = pd.DataFrame(issues)
    if issue_frame.empty:
        issue_frame = pd.DataFrame(
            columns=[
                "issue_id",
                "severity",
                "issue_type",
                "cik",
                "ticker",
                "field",
                "message",
                "source",
                "resolution_status",
            ]
        )
    issue_frame = issue_frame.drop_duplicates("issue_id")
    issue_frame.to_csv(issues_path, index=False)

    primary_results = [row for row in results if row["cik"] not in fallback_by_cik]
    effective_results = list({row["cik"]: row for row in results}.values())
    status_counts = Counter(row["status"] for row in effective_results)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_date": config["project"]["reference_date"],
        "method": (
            "Union of SEC Q4-2022 and Q1-2023 primary filings, one latest "
            "pre-reference or earliest post-reference filing per CIK; inline-XBRL "
            "Section 12(b) cover extraction. Current SEC ticker data are used for "
            "ticker-change context only and are not a historical-universe substitute."
        ),
        "sources": {
            "sec_q4_2022_index": {
                "url": universe_cfg["sec_filing_index_2022_q4_url"],
                "path": str(q4_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(q4_path),
            },
            "sec_q1_2023_index": {
                "url": universe_cfg["sec_filing_index_url"],
                "path": str(q1_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(q1_path),
            },
            "sec_current_tickers": {
                "url": universe_cfg["sec_current_tickers_url"],
                "path": str(tickers_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(tickers_path),
                "role": "subsequent ticker context only",
            },
        },
        "parameters": {
            "sec_requests_per_second": universe_cfg["sec_requests_per_second"],
            "sec_compressed_probe_bytes": probe_bytes,
            "workers": workers,
            "user_agent_source": universe_cfg["sec_user_agent_env"],
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "requests": requests.__version__,
            "pyyaml": yaml.__version__,
        },
        "counts": {
            "q4_primary_filings": len(q4),
            "q1_primary_filings": len(q1),
            "selected_ciks": len(selected),
            "alternate_filing_retries": len(fallback),
            "filing_probe_status": dict(status_counts),
            "evaluated_security_rows": len(universe),
            "included_securities": len(included),
            "issues": len(issue_frame),
        },
        "coverage_assessment": {
            "complete_historical_census_established": False,
            "coverage_percent": None,
            "reason": (
                "No accessible date-specific exchange directory was available. "
                "SEC filing/listing evidence yields a verifiable partial universe; "
                "the true count of all 3 January 2023 listed common equities is unknown."
            ),
        },
    }
    atomic_write_json(
        provenance, metadata_dir / "universe_source_provenance.json"
    )
    LOGGER.info(
        "Wrote %d evaluated rows and %d verified securities",
        len(universe),
        len(included),
    )
    return {
        "selected_ciks": len(selected),
        "evaluated_rows": len(universe),
        "included_securities": len(included),
        "issues": len(issue_frame),
        "universe_path": str(universe_path),
        "master_path": str(master_path),
        "issues_path": str(issues_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acquisition.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    result = run_build(config_path, force=args.force, workers=args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
