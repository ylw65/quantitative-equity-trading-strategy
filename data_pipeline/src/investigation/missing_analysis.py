"""Analyze securities without usable Yahoo Finance histories.

The analysis is deliberately descriptive. A failed Yahoo request is evidence
that the current vendor namespace did not return a usable history for the
requested symbol; it is not, by itself, evidence of delisting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import PROJECT_ROOT, atomic_write_json, configure_logging, load_config


LOGGER = logging.getLogger("us_equity_data.investigation.missing")
OUTPUT_DATA = PROJECT_ROOT / "data" / "investigation" / "missing"
OUTPUT_TABLES = PROJECT_ROOT / "reports" / "tables" / "investigation"
OUTPUT_FIGURES = PROJECT_ROOT / "reports" / "figures" / "investigation"
PROBE_DIR = OUTPUT_DATA / "targeted_probe"


def _string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"true", "1", "yes", "y"}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def classify_failure(error: str) -> str:
    """Classify the observed retrieval failure without inferring delisting."""
    text = _string(error).lower()
    if not text:
        return "no_error_recorded"
    if "too many requests" in text or "rate limit" in text:
        return "provider_rate_limit"
    if any(
        marker in text
        for marker in (
            "sslerror",
            "ssl:",
            "connection",
            "timed out",
            "timeout",
            "http 5",
        )
    ):
        return "provider_transport_error"
    if any(
        marker in text
        for marker in (
            "yftzmissingerror",
            "yfpricesmissingerror",
            "possibly delisted",
            "no timezone found",
            "no price data found",
            "quote not found",
        )
    ):
        return "provider_symbol_unavailable"
    if "http 403" in text or "forbidden" in text:
        return "provider_access_denied"
    return "other_provider_error"


def parse_change_targets(value: str) -> list[str]:
    """Extract candidate symbols from the SEC-derived ticker-change string."""
    text = _string(value)
    if "->" not in text:
        return []
    right = text.rsplit("->", 1)[1]
    return [
        token.strip().upper()
        for token in re.split(r"[,\s]+", right)
        if token.strip()
    ]


def _evidence_category(row: pd.Series) -> tuple[str, str]:
    """Return a mutually exclusive evidence category and its interpretation."""
    failure = classify_failure(_string(row.get("error")))
    if failure in {
        "provider_rate_limit",
        "provider_transport_error",
        "provider_access_denied",
        "other_provider_error",
    }:
        return failure, "The request did not establish symbol absence."

    current_ticker = _string(row.get("current_ticker")).upper()
    change_targets = parse_change_targets(row.get("subsequent_ticker_changes", ""))
    if current_ticker:
        return (
            "provider_no_history_for_current_listing",
            "SEC's current mapping lists a ticker, but Yahoo returned no history.",
        )
    if change_targets:
        return (
            "ticker_change_alias_unresolved",
            "SEC evidence indicates a subsequent symbol; old-symbol Yahoo lookup failed.",
        )
    if failure == "provider_symbol_unavailable":
        return (
            "provider_reports_symbol_missing_without_current_match",
            "Yahoo did not return a symbol history and SEC has no current exact ticker match.",
        )
    if failure == "no_error_recorded":
        return "unknown_no_usable_history", "No failure reason is available."
    return "unknown_no_usable_history", "Cause remains unresolved."


def _change_status(row: pd.Series) -> str:
    current = _string(row.get("current_ticker")).upper()
    targets = parse_change_targets(row.get("subsequent_ticker_changes", ""))
    if current:
        return "current_SEC_ticker_match"
    if targets:
        return "later_SEC_ticker_evidence"
    return "no_later_SEC_ticker_evidence"


def _normalise_inputs(
    master: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    if not master["security_id"].is_unique:
        raise ValueError("Security master contains duplicate security_id values.")
    if not manifest["security_id"].is_unique:
        raise ValueError("Manifest contains duplicate security_id values.")
    merged = master.merge(
        manifest.drop(
            columns=[
                "ticker_2023",
                "yahoo_ticker",
                "company_name",
                "exchange",
            ],
            errors="ignore",
        ),
        on="security_id",
        how="left",
        validate="one_to_one",
    )
    if merged["download_status"].isna().any():
        raise ValueError("Security master contains IDs absent from the manifest.")
    merged["download_status"] = merged["download_status"].fillna("not_attempted")
    merged["is_available"] = merged["download_status"].eq("success")
    merged["ticker_change_status"] = merged.apply(_change_status, axis=1)
    evidence = merged.apply(_evidence_category, axis=1)
    merged["evidence_category"] = [value[0] for value in evidence]
    merged["evidence_interpretation"] = [value[1] for value in evidence]
    merged["failure_category"] = merged["error"].map(classify_failure)
    merged["has_later_ticker_evidence"] = (
        merged["subsequent_ticker_changes"].map(parse_change_targets).map(bool)
    )
    merged["has_current_ticker"] = (
        merged["current_ticker"].map(_string).ne("")
    )
    merged["listing_date_known"] = merged["listing_date"].map(_string).ne("")
    merged["delisting_date_known"] = merged["delisting_date"].map(_string).ne("")
    merged["source_confidence"] = merged["source_confidence"].map(_string)
    return merged


def _group_comparison(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    grouped = (
        frame.assign(
            available=frame["is_available"].astype(int),
            unavailable=(~frame["is_available"]).astype(int),
        )
        .groupby(dimension, dropna=False)
        .agg(
            universe=("security_id", "size"),
            acquired=("available", "sum"),
            unavailable=("unavailable", "sum"),
        )
        .reset_index()
    )
    grouped["acquisition_rate"] = grouped["acquired"] / grouped["universe"]
    grouped["unavailable_rate"] = grouped["unavailable"] / grouped["universe"]
    grouped["universe_share"] = grouped["universe"] / len(frame)
    grouped["unavailable_share"] = (
        grouped["unavailable"] / max(1, int((~frame["is_available"]).sum()))
    )
    return grouped.sort_values(
        ["unavailable_rate", "unavailable"], ascending=[False, False]
    )


def _representative_cases(failed: pd.DataFrame) -> pd.DataFrame:
    candidates: list[pd.Series] = []
    for _, group in failed.groupby("evidence_category", sort=True):
        ordered = group.assign(
            known_change=group["has_later_ticker_evidence"].astype(int),
            known_current=group["has_current_ticker"].astype(int),
        ).sort_values(
            ["known_change", "known_current", "ticker_2023"],
            ascending=[False, False, True],
        )
        candidates.append(ordered.iloc[0])
    atvi = failed[failed["ticker_2023"].eq("ATVI")]
    if not atvi.empty:
        candidates.append(atvi.iloc[0])
    if not candidates:
        return pd.DataFrame()
    out = pd.DataFrame(candidates).drop_duplicates("security_id")
    out["change_targets"] = out["subsequent_ticker_changes"].map(
        lambda value: ", ".join(parse_change_targets(value))
    )
    return out[
        [
            "security_id",
            "ticker_2023",
            "company_name",
            "exchange",
            "security_type",
            "evidence_category",
            "failure_category",
            "ticker_change_status",
            "change_targets",
            "current_ticker",
            "error",
            "evidence_interpretation",
        ]
    ].reset_index(drop=True)


def _recovery_candidates(failed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in failed.iterrows():
        targets = parse_change_targets(row["subsequent_ticker_changes"])
        preferred_like = bool(
            len(targets) == 1
            and re.search(r"-P[A-Z0-9]*$", targets[0])
        )
        safe = (
            len(targets) == 1
            and bool(re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,9}", targets[0]))
            and targets[0] != _string(row["ticker_2023"]).upper()
            and not preferred_like
        )
        rows.append(
            {
                "security_id": row["security_id"],
                "ticker_2023": row["ticker_2023"],
                "company_name": row["company_name"],
                "exchange": row["exchange"],
                "security_type": row["security_type"],
                "subsequent_ticker_changes": row["subsequent_ticker_changes"],
                "candidate_ticker": targets[0] if len(targets) == 1 else "",
                "candidate_count": len(targets),
                "safe_single_alias": safe,
                "candidate_status": (
                    "candidate_for_identity_checked_probe"
                    if safe
                    else (
                        "preferred_or_noncommon_symbol_suspect"
                        if preferred_like
                        else (
                            "ambiguous_multi_symbol_mapping"
                            if len(targets) > 1
                            else "no_later_symbol_evidence"
                        )
                    )
                ),
                "evidence": (
                    "SEC current ticker reference maps the same CIK to one later symbol."
                    if safe
                    else (
                        "The later symbol has a preferred-security-like suffix and is not "
                        "an admissible common-equity alias."
                        if preferred_like
                        else "No unambiguous single later symbol in available SEC metadata."
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _probe_aliases(candidates: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return pd.DataFrame(
            columns=[
                "security_id",
                "ticker_2023",
                "candidate_ticker",
                "probe_status",
                "rows",
                "first_date",
                "last_date",
                "raw_probe_path",
                "error",
                "probe_interpretation",
            ]
        )
    import yfinance as yf

    chosen = candidates[candidates["safe_single_alias"]].head(limit)
    results: list[dict[str, Any]] = []
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in chosen.iterrows():
        started = time.monotonic()
        result = {
            "security_id": row["security_id"],
            "ticker_2023": row["ticker_2023"],
            "candidate_ticker": row["candidate_ticker"],
            "probe_status": "failed",
            "rows": 0,
            "first_date": "",
            "last_date": "",
            "raw_probe_path": "",
            "error": "",
        }
        try:
            data = yf.download(
                row["candidate_ticker"],
                start="2016-01-01",
                end="2026-01-01",
                auto_adjust=False,
                actions=True,
                repair=True,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=45,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(-1)
            data = data.reset_index()
            price_columns = [
                col
                for col in ("Open", "High", "Low", "Close", "Adj Close")
                if col in data.columns
            ]
            if price_columns and len(data):
                data = data.loc[
                    pd.to_numeric(
                        data[price_columns].stack(future_stack=True),
                        errors="coerce",
                    )
                    .unstack()
                    .notna()
                    .any(axis=1)
                ].copy()
            if len(data):
                date_col = "Date" if "Date" in data.columns else "Datetime"
                data["ticker"] = row["ticker_2023"]
                data["candidate_ticker"] = row["candidate_ticker"]
                data["security_id"] = row["security_id"]
                data["provider"] = "yahoo_finance_targeted_probe"
                path = PROBE_DIR / (
                    f"{row['ticker_2023']}__to__{row['candidate_ticker']}.parquet"
                )
                _atomic_parquet(data, path)
                result.update(
                    {
                        "probe_status": "vendor_data_exists_not_admitted",
                        "rows": len(data),
                        "first_date": str(pd.Timestamp(data[date_col].min()).date()),
                        "last_date": str(pd.Timestamp(data[date_col].max()).date()),
                        "raw_probe_path": str(path.relative_to(PROJECT_ROOT)),
                    }
                )
            else:
                result["probe_status"] = "no_usable_rows"
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
        if result["probe_status"] == "vendor_data_exists_not_admitted":
            if pd.Timestamp(result["first_date"]) > pd.Timestamp("2023-01-03"):
                result["probe_interpretation"] = (
                    "Vendor history begins after the reference date; continuity "
                    "across the historical symbol is unverified."
                )
            else:
                result["probe_interpretation"] = (
                    "Vendor history exists through the later SEC symbol; identity "
                    "continuity still requires admission checks."
                )
        else:
            result["probe_interpretation"] = "No usable vendor history returned."
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        result["probed_at_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        results.append(result)
        time.sleep(1.0)
    return pd.DataFrame(results)


def _figures(comparison: pd.DataFrame, failure_counts: pd.DataFrame) -> None:
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#6D7A80",
            "axes.linewidth": 0.7,
            "grid.color": "#DCE4E8",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    colors = ["#19324D", "#3F6F8F", "#4E8C86", "#B28A48", "#A45D4F", "#65727A"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, dimension, title in (
        (axes[0], "exchange", "Unavailable history by exchange"),
        (axes[1], "security_type", "Unavailable history by security type"),
    ):
        group = comparison[comparison["dimension"].eq(dimension)].copy()
        group = group.sort_values("unavailable_rate")
        ax.barh(
            group["value"].astype(str),
            group["unavailable_rate"],
            color=colors[: len(group)],
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Unavailable share within group")
        ax.set_title(title)
        for y, (_, row) in enumerate(group.iterrows()):
            ax.text(
                min(0.98, row["unavailable_rate"] + 0.015),
                y,
                f"{int(row['unavailable']):,}/{int(row['universe']):,}",
                va="center",
                fontsize=7,
            )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.18, wspace=0.55)
    fig.savefig(OUTPUT_FIGURES / "missing_by_exchange_type.pdf", format="pdf")
    plt.close(fig)

    categories = (
        failure_counts.groupby("evidence_category", dropna=False)["securities"]
        .sum()
        .rename("securities")
        .reset_index()
        .sort_values("securities")
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.barh(
        categories["evidence_category"].str.replace("_", " "),
        categories["securities"],
        color="#3F6F8F",
    )
    ax.set_xlabel("Unavailable securities")
    ax.set_title("Evidence categories for unavailable Yahoo histories")
    for y, value in enumerate(categories["securities"]):
        ax.text(value + 1, y, f"{int(value):,}", va="center", fontsize=8)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.90, bottom=0.20)
    fig.savefig(OUTPUT_FIGURES / "missing_failure_categories.pdf", format="pdf")
    plt.close(fig)


def run(
    *,
    master_path: str | Path,
    manifest_path: str | Path,
    probe_limit: int = 0,
) -> dict[str, Any]:
    configure_logging(PROJECT_ROOT / "logs" / "missing_security_investigation.log")
    master = pd.read_parquet(master_path)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    frame = _normalise_inputs(master, manifest)
    acquired = frame[frame["is_available"]].copy()
    failed = frame[~frame["is_available"]].copy()

    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    failure_counts = (
        failed.groupby(["failure_category", "evidence_category"], dropna=False)
        .size()
        .rename("securities")
        .reset_index()
    )
    failure_counts["share_of_unavailable"] = (
        failure_counts["securities"] / len(failed)
    )
    failure_counts = failure_counts.sort_values(
        ["securities", "failure_category"], ascending=[False, True]
    )

    comparisons = []
    for dimension in (
        "exchange",
        "security_type",
        "ticker_change_status",
        "has_current_ticker",
        "source_confidence",
        "listing_date_known",
        "delisting_date_known",
    ):
        group = _group_comparison(frame, dimension)
        group.insert(0, "dimension", dimension)
        group.insert(1, "value", group.pop(dimension).astype(str))
        comparisons.append(group)
    comparison = pd.concat(comparisons, ignore_index=True)

    representative = _representative_cases(failed)
    candidates = _recovery_candidates(failed)
    probes = _probe_aliases(candidates, probe_limit)
    if not probes.empty:
        candidates = candidates.merge(
            probes[
                [
                    "security_id",
                    "probe_status",
                    "rows",
                    "first_date",
                    "last_date",
                    "raw_probe_path",
                    "probe_interpretation",
                ]
            ],
            on="security_id",
            how="left",
        )
    candidates["admission_status"] = (
        "not_admitted_pending_identity_and_history_validation"
    )

    security_analysis = frame[
        [
            "security_id",
            "ticker_2023",
            "yahoo_ticker",
            "company_name",
            "exchange",
            "security_type",
            "permanent_id_type",
            "cik",
            "current_ticker",
            "subsequent_ticker_changes",
            "reference_date_verified",
            "source_confidence",
            "listing_date",
            "delisting_date",
            "download_status",
            "is_available",
            "failure_category",
            "evidence_category",
            "evidence_interpretation",
            "ticker_change_status",
        ]
    ].sort_values(["is_available", "ticker_2023"], ascending=[True, True])

    issues = pd.DataFrame(
        [
            {
                "issue_id": "MISS-001",
                "severity": "high",
                "issue_type": "unavailable_yahoo_histories",
                "affected_securities": len(failed),
                "status": "partially_resolved",
                "evidence": (
                    f"{len(failed)} of {len(frame)} verified securities have no "
                    "usable Yahoo history. Failure causes are recorded, but absence "
                    "does not establish that the securities never traded."
                ),
                "recommended_action": (
                    "Use targeted vendor/identity checks and retain coverage "
                    "uncertainty in all research samples."
                ),
            },
            {
                "issue_id": "MISS-002",
                "severity": "high",
                "issue_type": "nonrandom_missingness_risk",
                "affected_securities": int(
                    failed["ticker_change_status"]
                    .eq("later_SEC_ticker_evidence")
                    .sum()
                ),
                "status": "unresolved",
                "evidence": (
                    "Unavailable securities include SEC-backed later ticker "
                    "changes and are not distributed evenly by cohort attributes."
                ),
                "recommended_action": (
                    "Do not treat the acquired subset as representative; assess "
                    "missingness sensitivity after chronological splitting."
                ),
            },
            {
                "issue_id": "MISS-003",
                "severity": "medium",
                "issue_type": "targeted_alias_recovery_unadmitted",
                "affected_securities": int(candidates["safe_single_alias"].sum()),
                "status": "partially_resolved",
                "evidence": (
                    "Single-symbol SEC alias candidates were identified; any Yahoo "
                    "probe data remain outside the research dataset."
                ),
                "recommended_action": (
                    "Validate pre-change history under each alias before admitting it."
                ),
            },
        ]
    )

    paths = {
        "security": OUTPUT_DATA / "missing_security_analysis.csv",
        "failure": OUTPUT_DATA / "missing_failure_categories.csv",
        "comparison": OUTPUT_DATA / "missingness_by_exchange_type.csv",
        "representative": OUTPUT_DATA / "missing_representative_cases.csv",
        "candidates": OUTPUT_DATA / "targeted_recovery_candidates.csv",
        "probes": OUTPUT_DATA / "targeted_probe_results.csv",
        "issues": OUTPUT_DATA / "missing_security_issues.csv",
        "summary": OUTPUT_DATA / "missing_security_summary.json",
    }
    _atomic_csv(security_analysis, paths["security"])
    _atomic_csv(failure_counts, paths["failure"])
    _atomic_csv(comparison, paths["comparison"])
    _atomic_csv(representative, paths["representative"])
    _atomic_csv(candidates, paths["candidates"])
    _atomic_csv(probes, paths["probes"])
    _atomic_csv(issues, paths["issues"])
    table_exports = {
        "missing_failure_categories.csv": failure_counts,
        "missingness_by_exchange_type.csv": comparison,
        "missing_representative_cases.csv": representative,
        "targeted_recovery_candidates.csv": candidates,
        "missing_security_issues.csv": issues,
    }
    for filename, table in table_exports.items():
        _atomic_csv(table, OUTPUT_TABLES / filename)
    _figures(comparison, failure_counts)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe": len(frame),
        "acquired": len(acquired),
        "unavailable": len(failed),
        "acquisition_rate": len(acquired) / len(frame) if len(frame) else None,
        "failure_reason_counts": failure_counts.groupby("failure_category")[
            "securities"
        ].sum().to_dict(),
        "evidence_category_counts": failed["evidence_category"].value_counts().to_dict(),
        "ticker_change_status_counts": failed["ticker_change_status"].value_counts().to_dict(),
        "known_listing_dates": int(failed["listing_date_known"].sum()),
        "known_delisting_dates": int(failed["delisting_date_known"].sum()),
        "safe_single_alias_candidates": int(candidates["safe_single_alias"].sum()),
        "ambiguous_alias_candidates": int(
            candidates["candidate_status"]
            .eq("ambiguous_multi_symbol_mapping")
            .sum()
        ),
        "no_later_symbol_evidence": int(
            candidates["candidate_status"].eq("no_later_symbol_evidence").sum()
        ),
        "probe_attempts": len(probes),
        "probe_vendor_data_found": int(
            probes["probe_status"].eq("vendor_data_exists_not_admitted").sum()
        )
        if not probes.empty
        else 0,
        "probing_enabled": probe_limit > 0,
        "limitations": [
            "A provider miss is not proof of delisting or nonexistence.",
            "SEC listing/delisting date fields are incomplete.",
            "Alias probes are not admitted to the primary dataset automatically.",
            "The unavailable cohort is descriptive and not assumed representative.",
        ],
    }
    atomic_write_json(summary, paths["summary"])
    summary["files"] = {name: str(path) for name, path in paths.items()}
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        default="data/universe/security_master.parquet",
    )
    parser.add_argument(
        "--manifest",
        default="data/metadata/download_manifest.csv",
    )
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=0,
        help="Opt in to this many identity-checked Yahoo alias probes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        master_path=PROJECT_ROOT / args.master,
        manifest_path=PROJECT_ROOT / args.manifest,
        probe_limit=args.probe_limit,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
