"""Audit historical security identity for the fixed 488-security cohort.

This module is read-only with respect to every existing acquisition, validation,
filtering, and investigation artifact. It may populate a separate SEC metadata
cache under the audit output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import (
    PROJECT_ROOT,
    atomic_write_json,
    configure_logging,
    ensure_parent,
    load_config,
    project_path,
)
from src.universe.sec_client import SecClient


LOGGER = configure_logging(PROJECT_ROOT / "logs" / "security_identity_audit_488.log")

RELEVANT_FORMS = {
    "10-K",
    "10-Q",
    "20-F",
    "40-F",
    "8-K",
    "6-K",
    "S-1",
    "F-1",
    "S-4",
    "F-4",
    "425",
    "DEFM14A",
    "PREM14A",
    "424B3",
    "424B4",
    "424B5",
    "25",
    "25-NSE",
    "15-12B",
    "15-12G",
}

AUDIT_COLUMNS = [
    "security_id",
    "cik",
    "ticker_2023",
    "yahoo_ticker",
    "company_name",
    "exchange_2023",
    "first_yahoo_date",
    "last_yahoo_date",
    "acquisition_date",
    "current_ticker",
    "current_ticker_cik",
    "current_cik_match",
    "historical_event_detected",
    "ticker_reuse_detected",
    "economic_regime_change_detected",
    "listing_change_detected",
    "vendor_contamination_verified",
    "identity_audit_status",
    "evidence_confidence",
    "number_of_events",
    "number_of_unresolved_events",
    "review_priority",
    "review_notes",
]

BASELINE_EXTRA_COLUMNS = [
    "raw_path",
    "first_positive_volume_date",
    "last_positive_volume_date",
    "yahoo_row_count",
    "positive_volume_rows",
    "zero_volume_rows",
    "source_filing_date",
    "source_filing_url",
    "historical_eligibility_status",
    "historical_eligible_from",
    "current_entity_name",
    "current_ticker_issuer_name",
    "successor_name_similarity",
    "submission_cache_status",
    "relevant_filing_count",
    "relevant_forms",
    "event_verified",
]

EVENT_COLUMNS = [
    "event_id",
    "security_id",
    "cik",
    "ticker_before",
    "ticker_after",
    "issuer_before",
    "issuer_after",
    "issuer_cik_before",
    "issuer_cik_after",
    "event_type",
    "announcement_date",
    "effective_date",
    "first_trading_date",
    "exchange_before",
    "exchange_after",
    "security_class",
    "economic_continuity_status",
    "source_url",
    "source_filing_form",
    "source_filing_date",
    "evidence_excerpt",
    "verification_status",
]

REUSE_COLUMNS = [
    "ticker",
    "original_security_id",
    "original_issuer",
    "original_cik",
    "other_issuer",
    "other_cik",
    "original_last_verified_date",
    "other_first_verified_date",
    "overlap_or_gap_status",
    "reuse_evidence",
    "yahoo_checkpoint_affected",
    "potentially_affected_start",
    "potentially_affected_end",
    "contamination_status",
    "review_priority",
]

RECON_COLUMNS = [
    "security_id",
    "requested_cik",
    "requested_ticker",
    "vendor_series_id",
    "segment_start",
    "segment_end",
    "candidate_actual_issuer",
    "candidate_actual_cik",
    "identity_match_status",
    "supporting_evidence",
    "conflicting_evidence",
    "verification_status",
]

REVIEW_COLUMNS = [
    "security_id",
    "ticker",
    "company_name",
    "issue_type",
    "affected_period",
    "review_priority",
    "evidence_summary",
    "source_urls",
    "proposed_investigation",
    "recommended_treatment",
    "review_status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path, pattern: str = "*") -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob(pattern) if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".csv",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def cik10(value: Any) -> str:
    """Return a ten-character SEC CIK string without losing leading zeros."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nat", "<na>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10)


def normalize_issuer_name(value: str) -> str:
    import re

    text = str(value or "").upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    tokens = [
        token
        for token in text.split()
        if token
        not in {
            "INC",
            "INCORPORATED",
            "CORP",
            "CORPORATION",
            "COMPANY",
            "CO",
            "LTD",
            "LIMITED",
            "PLC",
            "SA",
            "HOLDINGS",
            "HOLDING",
        }
    ]
    return " ".join(tokens)


def issuer_name_similarity(first: str, second: str) -> float:
    return SequenceMatcher(
        None, normalize_issuer_name(first), normalize_issuer_name(second)
    ).ratio()


def _submission_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


def ensure_submission_cache(
    ciks: list[str],
    cache_dir: Path,
    *,
    requests_per_second: float,
    retries: int,
) -> pd.DataFrame:
    """Download only missing SEC submission records and return cache provenance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = SecClient(
        requests_per_second=requests_per_second,
        retries=retries,
    )
    rows = []
    for position, cik in enumerate(sorted(set(ciks)), start=1):
        target = cache_dir / f"CIK{cik}.json"
        status = "cached"
        if not target.exists():
            LOGGER.info(
                "Fetching SEC submissions %d/%d CIK %s",
                position,
                len(set(ciks)),
                cik,
            )
            client.get(_submission_url(cik), destination=target)
            status = "downloaded"
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception as error:
            raise RuntimeError(f"Unreadable SEC cache {target}: {error}") from error
        if cik10(payload.get("cik")) != cik:
            raise RuntimeError(f"SEC cache CIK mismatch in {target}")
        rows.append(
            {
                "cik": cik,
                "name": payload.get("name", ""),
                "url": _submission_url(cik),
                "cache_path": str(target),
                "status": status,
                "retrieved_at_utc": datetime.fromtimestamp(
                    target.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
                "sha256": _sha256(target),
            }
        )
    return pd.DataFrame(rows)


def submission_filings(payload: dict[str, Any]) -> pd.DataFrame:
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "accession_number": recent.get("accessionNumber", []),
            "filing_date": recent.get("filingDate", []),
            "report_date": recent.get("reportDate", []),
            "form": recent.get("form", []),
            "primary_document": recent.get("primaryDocument", []),
            "primary_document_description": recent.get(
                "primaryDocDescription", []
            ),
            "items": recent.get("items", []),
        }
    )


def _current_mapping(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["data"], columns=payload["fields"])
    frame["cik"] = frame["cik"].map(cik10)
    frame["ticker"] = frame["ticker"].fillna("").astype(str)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["exchange"] = frame["exchange"].fillna("").astype(str)
    by_cik = (
        frame.groupby("cik", sort=False)
        .agg(
            current_tickers=(
                "ticker",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
            current_names=(
                "name",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
            current_exchanges=(
                "exchange",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    by_ticker = (
        frame.sort_values(["ticker", "cik"])
        .drop_duplicates("ticker", keep="first")
        .rename(
            columns={
                "cik": "ticker_current_cik",
                "name": "ticker_current_name",
                "exchange": "ticker_current_exchange",
            }
        )
    )
    return by_cik, by_ticker


def _event_id(
    security_id: str, event_type: str, effective_date: Any, source_url: str
) -> str:
    digest = hashlib.sha256(
        f"{security_id}|{event_type}|{effective_date}|{source_url}".encode("utf-8")
    ).hexdigest()[:12].upper()
    return f"IEV-{digest}"


def _curated_events(
    evidence: dict[str, Any],
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    lookup = baseline.set_index("security_id")
    rows = []
    for evidence_ticker, block in evidence.items():
        if not isinstance(block, dict) or "events" not in block:
            continue
        security_id = block["security_id"]
        if security_id not in lookup.index:
            raise RuntimeError(f"Curated evidence security is not selected: {security_id}")
        for event in block["events"]:
            rows.append(
                {
                    "event_id": _event_id(
                        security_id,
                        event["event_type"],
                        event.get("effective_date", ""),
                        event.get("source_url", ""),
                    ),
                    "security_id": security_id,
                    "cik": cik10(lookup.at[security_id, "cik"]),
                    "ticker_before": event.get("ticker_before", ""),
                    "ticker_after": event.get("ticker_after", ""),
                    "issuer_before": event.get("issuer_before", ""),
                    "issuer_after": event.get("issuer_after", ""),
                    "issuer_cik_before": cik10(event.get("issuer_cik_before", "")),
                    "issuer_cik_after": cik10(event.get("issuer_cik_after", "")),
                    "event_type": event["event_type"],
                    "announcement_date": event.get("announcement_date", ""),
                    "effective_date": event.get("effective_date", ""),
                    "first_trading_date": event.get("first_trading_date", ""),
                    "exchange_before": event.get("exchange_before", ""),
                    "exchange_after": event.get("exchange_after", ""),
                    "security_class": event.get("security_class", ""),
                    "economic_continuity_status": event.get(
                        "economic_continuity_status", ""
                    ),
                    "source_url": event.get("source_url", ""),
                    "source_filing_form": event.get("source_filing_form", ""),
                    "source_filing_date": event.get("source_filing_date", ""),
                    "evidence_excerpt": event.get("evidence_excerpt", ""),
                    "verification_status": event.get("verification_status", ""),
                }
            )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _successor_events(
    baseline: pd.DataFrame,
    mapping_by_ticker: pd.DataFrame,
    *,
    similarity_threshold: float,
    curated_security_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    review_rows = []
    for row in baseline.itertuples(index=False):
        if row.security_id in curated_security_ids:
            continue
        if pd.isna(row.current_ticker_cik) or not row.current_ticker_cik:
            continue
        if str(row.current_ticker_cik) == str(row.cik):
            continue
        similarity = issuer_name_similarity(
            row.company_name, row.current_ticker_issuer_name
        )
        if similarity >= similarity_threshold:
            event_type = "legal_entity_successor_mapping_candidate"
            verification = "candidate_current_mapping_unresolved"
            continuity = "likely_economic_successor_unverified"
        else:
            event_type = "unrelated_issuer_current_ticker_mapping"
            verification = "verified_current_mapping"
            continuity = "different_issuer"
        if row.security_id == "SEC:0000009984":
            # Curated Barnes/Barrick evidence already represents this event more
            # precisely and avoids a duplicate generic record.
            continue
        rows.append(
            {
                "event_id": _event_id(
                    row.security_id,
                    event_type,
                    "",
                    "https://data.sec.gov/submissions/",
                ),
                "security_id": row.security_id,
                "cik": row.cik,
                "ticker_before": row.ticker_2023,
                "ticker_after": row.ticker_2023,
                "issuer_before": row.company_name,
                "issuer_after": row.current_ticker_issuer_name,
                "issuer_cik_before": row.cik,
                "issuer_cik_after": row.current_ticker_cik,
                "event_type": event_type,
                "announcement_date": "",
                "effective_date": "",
                "first_trading_date": "",
                "exchange_before": row.exchange_2023,
                "exchange_after": mapping_by_ticker.loc[
                    mapping_by_ticker["ticker"].eq(row.ticker_2023),
                    "ticker_current_exchange",
                ].iloc[0],
                "security_class": row.security_type,
                "economic_continuity_status": continuity,
                "source_url": "https://data.sec.gov/submissions/",
                "source_filing_form": "current ticker-to-CIK reference",
                "source_filing_date": row.current_mapping_access_date,
                "evidence_excerpt": (
                    f"Current SEC ticker mapping associates {row.ticker_2023} "
                    f"with CIK {row.current_ticker_cik}, not intended CIK {row.cik}."
                ),
                "verification_status": verification,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS), pd.DataFrame(review_rows)


def _raw_metrics(
    master: pd.DataFrame,
    identity_validation: pd.DataFrame,
    security_last_trade: pd.DataFrame,
) -> pd.DataFrame:
    needed = identity_validation[
        [
            "security_id",
            "raw_path",
            "n_rows",
            "positive_volume_rows",
            "first_valid_date",
            "last_valid_date",
        ]
    ].rename(
        columns={
            "n_rows": "yahoo_row_count",
            "first_valid_date": "raw_first_date",
            "last_valid_date": "raw_last_date",
        }
    )
    merged = master[["security_id"]].merge(
        needed, on="security_id", how="left", validate="one_to_one"
    )
    last = security_last_trade[
        ["security_id", "first_valid_date", "last_positive_volume_date"]
    ].rename(columns={"first_valid_date": "first_positive_volume_date"})
    merged = merged.merge(last, on="security_id", how="left", validate="one_to_one")
    merged["zero_volume_rows"] = (
        merged["yahoo_row_count"] - merged["positive_volume_rows"]
    )
    return merged


def _filing_inventory(
    master: pd.DataFrame,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = []
    payloads: dict[str, dict[str, Any]] = {}
    for row in master.itertuples(index=False):
        path = cache_dir / f"CIK{row.cik}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[row.cik] = payload
        filings = submission_filings(payload)
        if not filings.empty:
            filings["filing_date"] = pd.to_datetime(
                filings["filing_date"], errors="coerce"
            )
            filings = filings[
                filings["filing_date"].ge(pd.Timestamp("2015-01-01"))
            ]
            relevant = filings[filings["form"].isin(RELEVANT_FORMS)]
        else:
            relevant = filings
        form_counts = (
            relevant["form"].value_counts().sort_index().to_dict()
            if not relevant.empty
            else {}
        )
        rows.append(
            {
                "security_id": row.security_id,
                "cik": row.cik,
                "submission_name": payload.get("name", ""),
                "recent_filing_count": int(
                    len(payload.get("filings", {}).get("recent", {}).get("form", []))
                ),
                "relevant_filing_count": int(len(relevant)),
                "earliest_relevant_filing_date": (
                    relevant["filing_date"].min().date().isoformat()
                    if not relevant.empty
                    else ""
                ),
                "latest_relevant_filing_date": (
                    relevant["filing_date"].max().date().isoformat()
                    if not relevant.empty
                    else ""
                ),
                "relevant_forms": ";".join(form_counts),
                "form_counts_json": json.dumps(form_counts, sort_keys=True),
            }
        )
    return pd.DataFrame(rows), payloads


def _build_reconciliations(
    baseline: pd.DataFrame,
    evidence: dict[str, Any],
    raw: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for ticker in ("B", "FERG", "DKNG"):
        row = baseline.loc[baseline["ticker_2023"].eq(ticker)].iloc[0]
        raw_frame = raw[row["security_id"]]
        vendor_id = f"yahoo:{ticker}:{row['acquisition_date']}"
        if ticker == "DKNG":
            segments = [
                ("2019-07-25", "2020-04-23", "Diamond Eagle Acquisition Corp.", "0001772757", "same_listed_vehicle_spac_predecessor", "verified"),
                ("2020-04-24", "2025-12-31", "DraftKings Inc.", row["cik"], "intended_operating_company", "verified"),
            ]
        else:
            segments = [
                (
                    raw_frame["date"].min().date().isoformat(),
                    raw_frame["date"].max().date().isoformat(),
                    evidence[ticker]["vendor_reconciliation"]["candidate_actual_issuer"]
                    if "vendor_reconciliation" in evidence.get(ticker, {})
                    else row["company_name"],
                    evidence[ticker]["vendor_reconciliation"]["candidate_actual_cik"]
                    if "vendor_reconciliation" in evidence.get(ticker, {})
                    else row["cik"],
                    evidence[ticker]["vendor_reconciliation"]["identity_match_status"]
                    if "vendor_reconciliation" in evidence.get(ticker, {})
                    else "same_issuer_listing_change",
                    "verified",
                )
            ]
        for start, end, issuer, cik, status, verification in segments:
            rows.append(
                {
                    "security_id": row["security_id"],
                    "requested_cik": row["cik"],
                    "requested_ticker": ticker,
                    "vendor_series_id": vendor_id,
                    "segment_start": start,
                    "segment_end": end,
                    "candidate_actual_issuer": issuer,
                    "candidate_actual_cik": cik10(cik),
                    "identity_match_status": status,
                    "supporting_evidence": (
                        evidence[ticker]["vendor_reconciliation"][
                            "supporting_evidence"
                        ]
                        if ticker == "B"
                        else evidence[ticker]["events"][0]["evidence_excerpt"]
                    ),
                    "conflicting_evidence": (
                        evidence[ticker]["vendor_reconciliation"][
                            "conflicting_evidence"
                        ]
                        if ticker == "B"
                        else ""
                    ),
                    "verification_status": verification,
                }
            )

    # Current similar-name successor mappings are reviewed without asserting a
    # Yahoo contamination.
    candidates = baseline[
        baseline["current_cik_match"].eq("False")
        & baseline["successor_name_similarity"].ge(0.60)
    ]
    for row in candidates.itertuples(index=False):
        rows.append(
            {
                "security_id": row.security_id,
                "requested_cik": row.cik,
                "requested_ticker": row.ticker_2023,
                "vendor_series_id": f"yahoo:{row.yahoo_ticker}:{row.acquisition_date}",
                "segment_start": row.first_yahoo_date,
                "segment_end": row.last_yahoo_date,
                "candidate_actual_issuer": row.current_ticker_issuer_name,
                "candidate_actual_cik": row.current_ticker_cik,
                "identity_match_status": "successor_entity_candidate_unresolved",
                "supporting_evidence": (
                    "Current ticker maps to a similar-name successor legal entity; "
                    "economic and history continuity require review."
                ),
                "conflicting_evidence": (
                    "No direct evidence in the current inputs proves Yahoo combined "
                    "or substituted the two legal entities."
                ),
                "verification_status": "unresolved",
            }
        )
    return pd.DataFrame(rows, columns=RECON_COLUMNS)


def _build_reuse(
    baseline: pd.DataFrame,
    evidence: dict[str, Any],
    *,
    similarity_threshold: float,
) -> pd.DataFrame:
    rows = []
    candidates = baseline[
        baseline["current_cik_match"].eq("False")
        & baseline["successor_name_similarity"].lt(similarity_threshold)
    ]
    for row in candidates.itertuples(index=False):
        if row.ticker_2023 == "B":
            rows.append(
                {
                    "ticker": "B",
                    "original_security_id": row.security_id,
                    "original_issuer": row.company_name,
                    "original_cik": row.cik,
                    "other_issuer": row.current_ticker_issuer_name,
                    "other_cik": row.current_ticker_cik,
                    "original_last_verified_date": "2025-01-27",
                    "other_first_verified_date": "2025-05-09",
                    "overlap_or_gap_status": "gap_2025-01-28_to_2025-05-08",
                    "reuse_evidence": evidence["B"]["vendor_reconciliation"][
                        "supporting_evidence"
                    ],
                    "yahoo_checkpoint_affected": True,
                    "potentially_affected_start": row.first_yahoo_date,
                    "potentially_affected_end": row.last_yahoo_date,
                    "contamination_status": "vendor_contamination_verified",
                    "review_priority": "P0",
                }
            )
        else:
            rows.append(
                {
                    "ticker": row.ticker_2023,
                    "original_security_id": row.security_id,
                    "original_issuer": row.company_name,
                    "original_cik": row.cik,
                    "other_issuer": row.current_ticker_issuer_name,
                    "other_cik": row.current_ticker_cik,
                    "original_last_verified_date": "",
                    "other_first_verified_date": "",
                    "overlap_or_gap_status": "unresolved",
                    "reuse_evidence": (
                        "Current ticker maps to a different CIK and issuer name."
                    ),
                    "yahoo_checkpoint_affected": "unknown",
                    "potentially_affected_start": "",
                    "potentially_affected_end": "",
                    "contamination_status": "possible_not_verified",
                    "review_priority": "P1",
                }
            )
    return pd.DataFrame(rows, columns=REUSE_COLUMNS)


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(project_path(config_path))
    inputs = config["inputs"]
    output_root = project_path(config["outputs"]["root"])
    cache_dir = project_path(config["outputs"]["sec_submissions_cache"])
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    protected_files = [
        project_path(inputs["selected_master"]),
        project_path(inputs["selected_daily"]),
        project_path(inputs["download_manifest"]),
        project_path(inputs["security_last_trade"]),
        project_path(inputs["identity_validation"]),
        project_path(inputs["current_sec_tickers"]),
        project_path(inputs["historical_eligibility_policy"]),
        project_path(inputs["curated_evidence"]),
    ]
    hashes_before = {str(path): _sha256(path) for path in protected_files}
    raw_root = project_path(inputs["raw_yahoo_root"])
    raw_tree_before = tree_sha256(raw_root)

    master = pd.read_parquet(project_path(inputs["selected_master"]))
    master["security_id"] = master["security_id"].astype(str)
    master["cik"] = master["cik"].map(cik10)
    master["exchange_2023"] = master["exchange"]
    if len(master) != 488 or master["security_id"].nunique() != 488:
        raise RuntimeError("Selected master is not the expected 488 unique securities")
    manifest = pd.read_csv(
        project_path(inputs["download_manifest"]), low_memory=False
    )
    manifest["security_id"] = manifest["security_id"].astype(str)
    manifest = manifest[manifest["security_id"].isin(set(master["security_id"]))]
    identity_validation = pd.read_csv(
        project_path(inputs["identity_validation"]), low_memory=False
    )
    identity_validation["security_id"] = identity_validation[
        "security_id"
    ].astype(str)
    security_last_trade = pd.read_csv(
        project_path(inputs["security_last_trade"]), low_memory=False
    )
    security_last_trade["security_id"] = security_last_trade[
        "security_id"
    ].astype(str)
    policy = pd.read_csv(
        project_path(inputs["historical_eligibility_policy"]),
        dtype=str,
        low_memory=False,
    )
    evidence = json.loads(
        project_path(inputs["curated_evidence"]).read_text(encoding="utf-8")
    )
    current_path = project_path(inputs["current_sec_tickers"])
    by_cik, by_ticker = _current_mapping(current_path)
    current_access_date = datetime.fromtimestamp(
        current_path.stat().st_mtime, timezone.utc
    ).date().isoformat()

    baseline = master.merge(
        manifest[
            [
                "security_id",
                "download_status",
                "completed_at_utc",
                "attempted_at_utc",
            ]
        ],
        on="security_id",
        how="left",
        validate="one_to_one",
    )
    raw_metrics = _raw_metrics(master, identity_validation, security_last_trade)
    baseline = baseline.merge(
        raw_metrics, on="security_id", how="left", validate="one_to_one"
    ).merge(by_cik, on="cik", how="left", validate="many_to_one").merge(
        by_ticker[
            [
                "ticker",
                "ticker_current_cik",
                "ticker_current_name",
                "ticker_current_exchange",
            ]
        ],
        left_on="ticker_2023",
        right_on="ticker",
        how="left",
        validate="many_to_one",
    )
    baseline["acquisition_date"] = baseline["completed_at_utc"].fillna(
        baseline["attempted_at_utc"]
    )
    baseline["first_yahoo_date"] = baseline["raw_first_date"]
    baseline["last_yahoo_date"] = baseline["raw_last_date"]
    baseline["current_ticker_cik"] = baseline["ticker_current_cik"].fillna("")
    baseline["current_ticker_issuer_name"] = baseline["ticker_current_name"].fillna("")
    baseline["current_entity_name"] = baseline["current_names"].fillna("")
    baseline["current_ticker"] = baseline["current_tickers"].fillna("")
    baseline["current_cik_match"] = np.where(
        baseline["ticker_current_cik"].isna(),
        "",
        baseline["ticker_current_cik"].eq(baseline["cik"]).map(
            {True: "True", False: "False"}
        ),
    )
    baseline["successor_name_similarity"] = [
        issuer_name_similarity(first, second)
        if second
        else np.nan
        for first, second in zip(
            baseline["company_name"],
            baseline["current_ticker_issuer_name"],
            strict=True,
        )
    ]

    policy_lookup = policy.set_index("security_id")
    baseline["historical_eligibility_status"] = baseline["security_id"].map(
        lambda value: (
            policy_lookup.at[value, "policy_type"]
            if value in policy_lookup.index
            else "default_2016_2025_history"
        )
    )
    baseline["historical_eligible_from"] = baseline["security_id"].map(
        lambda value: (
            policy_lookup.at[value, "eligible_from"]
            if value in policy_lookup.index
            else ""
        )
    )
    baseline["source_filing_date"] = baseline["filing_date"]
    baseline["source_filing_url"] = baseline["filing_url"]
    baseline["current_mapping_access_date"] = current_access_date

    all_ciks = list(
        dict.fromkeys(
            [
                *baseline["cik"].tolist(),
                *[
                    value
                    for value in baseline["ticker_current_cik"].dropna().tolist()
                    if value and value not in set(baseline["cik"])
                ],
            ]
        )
    )
    cache_manifest = ensure_submission_cache(
        all_ciks,
        cache_dir,
        requests_per_second=float(config["project"]["sec_requests_per_second"]),
        retries=int(config["project"]["sec_retries"]),
    )
    filing_inventory, _ = _filing_inventory(baseline, cache_dir)
    baseline = baseline.merge(
        filing_inventory[
            [
                "security_id",
                "relevant_filing_count",
                "relevant_forms",
            ]
        ],
        on="security_id",
        how="left",
        validate="one_to_one",
    )
    baseline["submission_cache_status"] = "cached"

    event_frames = [_curated_events(evidence, baseline)]
    successor_events, _ = _successor_events(
        baseline,
        by_ticker,
        similarity_threshold=float(
            config["project"]["successor_name_similarity_threshold"]
        ),
        curated_security_ids={
            block["security_id"]
            for block in evidence.values()
            if isinstance(block, dict) and "security_id" in block
        },
    )
    event_frames.append(successor_events)
    events = pd.concat(event_frames, ignore_index=True)
    events = events.drop_duplicates("event_id").reset_index(drop=True)

    raw_frames = {}
    for ticker in ("B", "FERG", "DKNG"):
        row = baseline.loc[baseline["ticker_2023"].eq(ticker)].iloc[0]
        raw_frames[row["security_id"]] = pd.read_parquet(
            project_path(row["raw_path"])
        )
    reconciliation = _build_reconciliations(baseline, evidence, raw_frames)
    reuse = _build_reuse(
        baseline,
        evidence,
        similarity_threshold=float(
            config["project"]["successor_name_similarity_threshold"]
        ),
    )

    event_counts = events.groupby("security_id", sort=False).agg(
        number_of_events=("event_id", "size"),
        number_of_unresolved_events=(
            "verification_status",
            lambda values: int(
                pd.Series(values)
                .astype(str)
                .str.contains(
                    "unresolved|candidate|filing_record_only",
                    case=False,
                    regex=True,
                )
                .sum()
            ),
        ),
    )
    baseline = baseline.merge(
        event_counts, on="security_id", how="left", validate="one_to_one"
    ).fillna({"number_of_events": 0, "number_of_unresolved_events": 0})
    baseline["number_of_events"] = baseline["number_of_events"].astype(int)
    baseline["number_of_unresolved_events"] = baseline[
        "number_of_unresolved_events"
    ].astype(int)
    baseline["historical_event_detected"] = baseline["number_of_events"].gt(0)
    baseline["ticker_reuse_detected"] = baseline["security_id"].eq(
        "SEC:0000009984"
    )
    has_regime = events["economic_continuity_status"].astype(str).str.contains(
        "regime|reorganization|successor|spac|different_issuer", case=False
    ) | events["event_type"].astype(str).str.contains(
        "ticker_reused|legal_entity_reorganization", case=False
    )
    regime_ids = set(events.loc[has_regime, "security_id"])
    baseline["economic_regime_change_detected"] = baseline["security_id"].isin(
        regime_ids
    )
    has_listing = events["event_type"].astype(str).str.contains(
        "listing|delisting", case=False
    )
    listing_ids = set(events.loc[has_listing, "security_id"])
    baseline["listing_change_detected"] = baseline["security_id"].isin(listing_ids)
    baseline["vendor_contamination_verified"] = baseline["security_id"].eq(
        "SEC:0000009984"
    )
    verified_event_ids = set(
        events.loc[
            ~events["verification_status"]
            .astype(str)
            .str.contains("candidate|unresolved|filing_record_only", case=False),
            "security_id",
        ]
    )
    baseline["event_verified"] = baseline["security_id"].isin(verified_event_ids)
    baseline["identity_audit_status"] = "no_conflict_detected"
    baseline.loc[
        baseline["security_id"].eq("SEC:0000009984"), "identity_audit_status"
    ] = "vendor_series_mismatch"
    baseline.loc[
        baseline["ticker_2023"].eq("FERG"), "identity_audit_status"
    ] = "listing_eligibility_change"
    baseline.loc[
        baseline["ticker_2023"].eq("DKNG"), "identity_audit_status"
    ] = "economic_regime_change"
    baseline.loc[
        baseline["security_id"].isin(
            set(
                events.loc[
                    events["event_type"].eq("legal_entity_reorganization"),
                    "security_id",
                ]
            )
            - {"SEC:0001832433"}
        ),
        "identity_audit_status",
    ] = "economic_regime_change"
    unresolved_ids = set(
        baseline.loc[
            baseline["number_of_unresolved_events"].gt(0), "security_id"
        ]
    )
    baseline.loc[
        baseline["security_id"].isin(unresolved_ids)
        & ~baseline["identity_audit_status"].eq("vendor_series_mismatch"),
        "identity_audit_status",
    ] = "unresolved_identity"
    baseline.loc[
        baseline["ticker_2023"].eq("FERG"), "identity_audit_status"
    ] = "listing_eligibility_change"
    baseline["evidence_confidence"] = np.where(
        baseline["identity_audit_status"].eq("no_conflict_detected"),
        "low",
        "high",
    )
    baseline["review_priority"] = np.select(
        [
            baseline["vendor_contamination_verified"],
            baseline["ticker_reuse_detected"],
            baseline["number_of_unresolved_events"].gt(0),
            baseline["historical_event_detected"],
        ],
        ["P0", "P0", "P1", "P2"],
        default="P3",
    )
    baseline["review_notes"] = np.select(
        [
            baseline["vendor_contamination_verified"],
            baseline["ticker_reuse_detected"],
            baseline["number_of_unresolved_events"].gt(0),
            baseline["historical_event_detected"],
        ],
        [
            "Yahoo series contains the wrong issuer; no observations were changed.",
            "Ticker was reused after the intended issuer ceased public trading.",
            "Current mapping or legal-successor continuity requires human review.",
            "Verified historical event; no automatic data treatment applied.",
        ],
        default=(
            "No conflict found in current SEC mapping or curated evidence; "
            "continuous historical identity is not independently proven."
        ),
    )

    audit = baseline[AUDIT_COLUMNS + BASELINE_EXTRA_COLUMNS].copy()
    review_rows = []
    for row in audit[audit["review_priority"].ne("P3")].itertuples(index=False):
        security_events = events[events["security_id"].eq(row.security_id)]
        source_urls = ";".join(
            sorted(set(security_events["source_url"].dropna().astype(str)))
        )
        review_rows.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker_2023,
                "company_name": row.company_name,
                "issue_type": row.identity_audit_status,
                "affected_period": (
                    f"{row.first_yahoo_date} to {row.last_yahoo_date}"
                ),
                "review_priority": row.review_priority,
                "evidence_summary": row.review_notes,
                "source_urls": source_urls,
                "proposed_investigation": (
                    "Resolve legal/economic continuity and compare the Yahoo "
                    "checkpoint against an independent historical identifier map."
                ),
                "recommended_treatment": "pending_decision",
                "review_status": "pending_decision",
            }
        )
    review = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS).sort_values(
        ["review_priority", "ticker", "security_id"]
    )

    _atomic_csv(audit, output_root / "security_identity_audit_488.csv")
    _atomic_csv(events, output_root / "historical_security_events.csv")
    _atomic_csv(reuse, output_root / "ticker_reuse_candidates.csv")
    _atomic_csv(reconciliation, output_root / "vendor_series_reconciliation.csv")
    _atomic_csv(review, output_root / "identity_review_register.csv")
    _atomic_csv(cache_manifest, output_root / "sec_submissions_cache_manifest.csv")
    _atomic_csv(filing_inventory, output_root / "sec_filing_inventory.csv")

    counts = audit["identity_audit_status"].value_counts().to_dict()
    verified_reuse = int(
        reuse["contamination_status"].eq("vendor_contamination_verified").sum()
    )
    vendor_errors = int(audit["vendor_contamination_verified"].sum())
    unresolved = int(audit["identity_audit_status"].eq("unresolved_identity").sum())
    ticker_change_ids = set(
        events.loc[
            events["ticker_before"].fillna("").astype(str)
            != events["ticker_after"].fillna("").astype(str),
            "security_id",
        ]
    )
    summary = {
        "generated_at_utc": utc_now(),
        "audit_date": config["project"]["audit_date"],
        "securities_audited": int(len(audit)),
        "no_conflict_detected": int(counts.get("no_conflict_detected", 0)),
        "potential_identity_problems": int(
            audit["identity_audit_status"].ne("no_conflict_detected").sum()
        ),
        "ticker_changes": len(ticker_change_ids),
        "same_issuer_ticker_changes": int(
            counts.get("ticker_change_same_issuer", 0)
        ),
        "potential_ticker_reuse": int(len(reuse)),
        "verified_ticker_reuse": verified_reuse,
        "economic_regime_changes": int(
            audit["economic_regime_change_detected"].sum()
        ),
        "listing_changes": int(audit["listing_change_detected"].sum()),
        "verified_yahoo_attribution_errors": vendor_errors,
        "unresolved_identity": unresolved,
        "human_review": int(len(review)),
        "status_counts": counts,
        "highest_priority": review.head(10).to_dict("records"),
        "barnes_barrick": {
            "ticker_reuse_verified": True,
            "vendor_contamination_verified": True,
            "intended_issuer": "Barnes Group Inc. / CIK 0000009984",
            "returned_series_issuer": "Barrick Mining Corp. / CIK 0000756894",
            "barnes_public_equity_end": "2025-01-27",
            "barrick_b_ticker_start": "2025-05-09",
        },
        "limitations": [
            "SEC filings are not a complete daily ticker directory.",
            "No historical Yahoo prices were downloaded for this audit.",
            "Current SEC mappings are a snapshot and do not establish continuity.",
            "No observations were corrected, excluded, or reassigned.",
        ],
    }
    atomic_write_json(summary, output_root / "identity_audit_summary.json")

    summary_lines = [
        "# Historical Security Identity Audit for the 488-Stock Dataset",
        "",
        "This is a read-only audit. No Yahoo checkpoint, validated observation,",
        "liquidity-screen file, selected master, or existing quality flag was changed.",
        "",
        "## Summary",
        "",
        f"- Securities audited: {len(audit):,}",
        f"- No conflict detected: {counts.get('no_conflict_detected', 0):,}",
        f"- Potential identity problems: {audit['identity_audit_status'].ne('no_conflict_detected').sum():,}",
        f"- Securities with ticker-change events: {len(ticker_change_ids):,}",
        f"- Same-issuer ticker changes: {counts.get('ticker_change_same_issuer', 0):,}",
        f"- Potential ticker-reuse relationships: {len(reuse):,}",
        f"- Verified ticker-reuse relationships: {verified_reuse:,}",
        f"- Economic-regime changes: {audit['economic_regime_change_detected'].sum():,}",
        f"- Listing changes: {audit['listing_change_detected'].sum():,}",
        f"- Verified Yahoo attribution errors: {vendor_errors:,}",
        f"- Unresolved identity concerns: {unresolved:,}",
        f"- Cases in the human-review register: {len(review):,}",
        "",
        "## Highest-priority cases",
        "",
        "| Priority | Ticker | Issuer | Status |",
        "|---|---|---|---|",
    ]
    for row in review.head(10).itertuples(index=False):
        summary_lines.append(
            f"| {row.review_priority} | {row.ticker} | {row.company_name} | "
            f"{row.issue_type} |"
        )
    summary_lines.extend(
        [
            "",
            "## Barnes and Barrick",
            "",
            "The intended B security is Barnes Group Inc., CIK 0000009984.",
            "Barnes ceased public trading on 2025-01-27. Barrick Mining Corp.,",
            "CIK 0000756894, adopted NYSE ticker B at the start of trading on",
            "2025-05-09. The Yahoo checkpoint associated with Barnes contains a",
            "continuous series through 2025, closes near USD 16 on the Barnes",
            "delisting date rather than the verified USD 47.50 cash consideration,",
            "and continues after Barnes ceased trading. The checkpoint is therefore",
            "classified as a verified wrong-issuer vendor series. No correction was applied.",
            "",
            "## Limitations",
            "",
            "SEC filing metadata and current ticker references are not a complete",
            "dated exchange directory. An unchanged current CIK mapping does not",
            "prove uninterrupted historical identity. The absence of a detected",
            "conflict is not a positive verification of every historical observation.",
        ]
    )
    (output_root / "identity_audit_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    hashes_after = {str(path): _sha256(path) for path in protected_files}
    raw_tree_after = tree_sha256(raw_root)
    if hashes_before != hashes_after or raw_tree_before != raw_tree_after:
        raise RuntimeError("A protected research input changed during the audit")
    summary["protected_input_hashes_sha256"] = hashes_after
    summary["raw_yahoo_tree_sha256"] = raw_tree_after
    summary["protected_inputs_unchanged"] = True
    summary["software"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    atomic_write_json(summary, output_root / "identity_audit_summary.json")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "security_identity_audit.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
