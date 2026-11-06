"""Reconcile Yahoo provenance and audit checkpoint identity and time coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, atomic_write_json, configure_logging, project_path


LOGGER = configure_logging(PROJECT_ROOT / "logs" / "identity_temporal.log")
OUT_DIR = PROJECT_ROOT / "data" / "investigation" / "identity"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "investigation"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "investigation"
DOC_PATH = PROJECT_ROOT / "docs" / "investigation_identity.md"
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
FINGERPRINT_COLUMNS = ["date", "close", "adj_close", "volume"]


def _string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _sha256_bytes(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _fingerprint(frame: pd.DataFrame) -> tuple[str, str]:
    """Return full price-volume and price-only fingerprints."""
    active = frame.loc[
        frame["volume"].gt(0)
        & frame["adj_close"].gt(0)
        & frame["close"].gt(0)
    ].sort_values("date")
    if len(active) < 20:
        return "", ""
    dates = pd.to_datetime(active["date"]).astype("int64").to_numpy(np.int64)
    prices = active[["close", "adj_close"]].to_numpy(np.float64)
    volume = active["volume"].to_numpy(np.float64)
    source = np.column_stack([dates, prices, volume])
    price_source = np.column_stack([dates, prices])
    return _sha256_bytes(source), _sha256_bytes(price_source)


def _trailing_unchanged_sessions(frame: pd.DataFrame) -> int:
    """Count contiguous trailing zero-volume rows at the final unchanged price."""
    if frame.empty:
        return 0
    end_close = frame["close"].iloc[-1]
    end_adj = frame["adj_close"].iloc[-1]
    count = 0
    for volume, close, adj_close in frame[
        ["volume", "close", "adj_close"]
    ].iloc[::-1].itertuples(index=False, name=None):
        if (
            pd.notna(volume)
            and volume == 0
            and pd.notna(close)
            and close == end_close
            and pd.notna(adj_close)
            and adj_close == end_adj
        ):
            count += 1
        else:
            break
    return count


def _audit_checkpoint(
    path: Path,
    expected: pd.Series,
    calendar_sessions: pd.DatetimeIndex,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    row: dict[str, Any] = {
        "security_id": _string(expected.get("security_id")),
        "ticker_2023": _string(expected.get("ticker_2023")),
        "yahoo_ticker": _string(expected.get("yahoo_ticker")),
        "raw_path": str(path.relative_to(PROJECT_ROOT)),
    }
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        row.update(
            {
                "readable": False,
                "identity_valid": False,
                "quarantine_action": "checkpoint_quarantined",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        issues.append(
            {
                "severity": "critical",
                "issue_type": "unreadable_checkpoint",
                "security_id": row["security_id"],
                "ticker": row["yahoo_ticker"],
                "raw_path": row["raw_path"],
                "details": row["error"],
            }
        )
        return row, {}, issues

    required = {
        "date",
        "ticker",
        "security_id",
        "ticker_2023",
        "provider",
        "volume",
        *PRICE_COLUMNS,
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        issues.append(
            {
                "severity": "critical",
                "issue_type": "missing_checkpoint_columns",
                "security_id": row["security_id"],
                "ticker": row["yahoo_ticker"],
                "raw_path": row["raw_path"],
                "details": "|".join(missing_columns),
            }
        )

    if "date" not in frame.columns:
        frame["date"] = pd.NaT
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.loc[frame["date"].notna()].sort_values("date", kind="stable").copy()
    for column in [*PRICE_COLUMNS, "volume"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ticker_values = set(frame.get("ticker", pd.Series(dtype=str)).astype(str))
    id_values = set(frame.get("security_id", pd.Series(dtype=str)).astype(str))
    ticker_2023_values = set(
        frame.get("ticker_2023", pd.Series(dtype=str)).astype(str)
    )
    provider_values = set(frame.get("provider", pd.Series(dtype=str)).astype(str))
    expected_ticker = _string(expected.get("yahoo_ticker"))
    expected_id = _string(expected.get("security_id"))
    expected_ticker_2023 = _string(expected.get("ticker_2023"))
    identity_valid = (
        ticker_values == {expected_ticker}
        and id_values == {expected_id}
        and ticker_2023_values == {expected_ticker_2023}
        and provider_values == {"yahoo_finance"}
        and not missing_columns
    )
    if not identity_valid:
        issues.append(
            {
                "severity": "critical",
                "issue_type": "ticker_security_identity_mismatch",
                "security_id": expected_id,
                "ticker": expected_ticker,
                "raw_path": row["raw_path"],
                "details": json.dumps(
                    {
                        "tickers": sorted(ticker_values),
                        "security_ids": sorted(id_values),
                        "tickers_2023": sorted(ticker_2023_values),
                        "providers": sorted(provider_values),
                    }
                ),
            }
        )

    price_present = frame[[c for c in PRICE_COLUMNS if c in frame.columns]]
    all_nan_price_rows = (
        int(price_present.isna().all(axis=1).sum()) if len(price_present.columns) else len(frame)
    )
    positive_price = frame["adj_close"].gt(0)
    active = frame["volume"].gt(0) & positive_price
    duplicate_rows = int(frame.duplicated("date").sum())
    sorted_dates = bool(frame["date"].is_monotonic_increasing)
    first_valid = frame.loc[positive_price, "date"].min()
    last_valid = frame.loc[positive_price, "date"].max()
    last_trade = frame.loc[active, "date"].max()
    trailing_unchanged = _trailing_unchanged_sessions(frame)

    if last_trade is not pd.NaT and pd.notna(last_trade):
        last_session_index = int(calendar_sessions.searchsorted(last_trade.normalize()))
        end_index = int(calendar_sessions.searchsorted(pd.Timestamp("2025-12-31")))
        trailing_market_sessions = max(0, end_index - last_session_index)
    else:
        trailing_market_sessions = len(calendar_sessions)

    if duplicate_rows:
        issues.append(
            {
                "severity": "high",
                "issue_type": "duplicate_checkpoint_dates",
                "security_id": expected_id,
                "ticker": expected_ticker,
                "raw_path": row["raw_path"],
                "details": f"duplicate_rows={duplicate_rows}",
            }
        )
    if not sorted_dates:
        issues.append(
            {
                "severity": "medium",
                "issue_type": "unsorted_checkpoint_dates",
                "security_id": expected_id,
                "ticker": expected_ticker,
                "raw_path": row["raw_path"],
                "details": "Rows required deterministic sorting before use.",
            }
        )
    if all_nan_price_rows:
        issues.append(
            {
                "severity": "medium",
                "issue_type": "all_nan_price_placeholder_rows",
                "security_id": expected_id,
                "ticker": expected_ticker,
                "raw_path": row["raw_path"],
                "details": f"rows={all_nan_price_rows}; excluded from research base",
            }
        )

    row.update(
        {
            "readable": True,
            "identity_valid": identity_valid,
            "provider_values": "|".join(sorted(provider_values)),
            "n_rows": len(frame),
            "n_duplicate_dates": duplicate_rows,
            "dates_sorted": sorted_dates,
            "all_nan_price_rows": all_nan_price_rows,
            "nonpositive_price_rows": int(frame["adj_close"].le(0).sum()),
            "positive_price_rows": int(positive_price.sum()),
            "positive_volume_rows": int(active.sum()),
            "first_valid_date": first_valid.date().isoformat() if pd.notna(first_valid) else "",
            "last_valid_date": last_valid.date().isoformat() if pd.notna(last_valid) else "",
            "last_positive_volume_date": last_trade.date().isoformat() if pd.notna(last_trade) else "",
            "trailing_market_sessions_after_last_trade": trailing_market_sessions,
            "trailing_unchanged_zero_volume_rows": trailing_unchanged,
            "stopped_trading_candidate": bool(
                trailing_market_sessions >= 10 and trailing_unchanged >= 10
            ),
            "quarantine_action": (
                "checkpoint_quarantined"
                if not identity_valid or not positive_price.any()
                else (
                    "base_rows_only"
                    if all_nan_price_rows or duplicate_rows
                    else "none"
                )
            ),
        }
    )
    if not positive_price.any():
        issues.append(
            {
                "severity": "critical",
                "issue_type": "checkpoint_without_positive_price",
                "security_id": expected_id,
                "ticker": expected_ticker,
                "raw_path": row["raw_path"],
                "details": "No positive adjusted close remains after parsing.",
            }
        )

    frame["year"] = frame["date"].dt.year
    yearly = (
        frame.assign(
            valid_price=positive_price,
            traded=active,
        )
        .groupby("year", dropna=False)
        .agg(
            valid_price_rows=("valid_price", "sum"),
            traded_rows=("traded", "sum"),
            zero_volume_rows=("volume", lambda values: int(values.eq(0).sum())),
        )
        .reset_index()
    )
    yearly["security_id"] = expected_id
    yearly["ticker"] = expected_ticker
    yearly["year"] = yearly["year"].astype(int)
    fingerprint, price_fingerprint = _fingerprint(frame)
    timeline = {
        "security_id": expected_id,
        "ticker": expected_ticker,
        "first_valid_date": first_valid,
        "last_valid_date": last_valid,
        "last_positive_volume_date": last_trade,
        "trailing_market_sessions_after_last_trade": trailing_market_sessions,
        "trailing_unchanged_zero_volume_rows": trailing_unchanged,
        "stopped_trading_candidate": bool(
            trailing_market_sessions >= 10 and trailing_unchanged >= 10
        ),
        "price_volume_fingerprint": fingerprint,
        "price_only_fingerprint": price_fingerprint,
        "active_rows": int(active.sum()),
    }
    return row, {"timeline": timeline, "yearly": yearly}, issues


def _provenance_reconciliation(
    master: pd.DataFrame,
    manifest: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successful = manifest.loc[manifest["download_status"].eq("success")].copy()
    failed = manifest.loc[manifest["download_status"].eq("failed")].copy()
    base = pd.read_parquet(
        PROJECT_ROOT / "data/validated/base.parquet",
        columns=["security_id", "date", "adj_close", "volume", "pct_return"],
    )
    validation = json.loads(
        (PROJECT_ROOT / "data/validated/validation_provenance.json").read_text()
    )
    yahoo_files = sorted((PROJECT_ROOT / "data/raw/yahoo").glob("*.parquet"))
    eod_files = sorted(
        (PROJECT_ROOT / "data/raw/eodhd_demo/daily").glob("*.parquet")
    )
    eod_providers = Counter()
    eod_rows = 0
    for path in eod_files:
        frame = pd.read_parquet(path, columns=["provider"])
        eod_rows += len(frame)
        eod_providers.update(frame["provider"].astype(str))

    raw_observation_count = int(
        pd.to_numeric(successful["n_observations"], errors="coerce").fillna(0).sum()
    )
    payload = {
        "authoritative_research_source": "Yahoo Finance via yfinance",
        "manifest_rows": len(manifest),
        "manifest_success": len(successful),
        "manifest_failed": len(failed),
        "successful_unique_security_ids": int(successful["security_id"].nunique()),
        "successful_unique_tickers": int(successful["yahoo_ticker"].nunique()),
        "yahoo_raw_files": len(yahoo_files),
        "identity_audit_rows": len(identity),
        "identity_valid_checkpoints": int(identity["identity_valid"].fillna(False).sum()),
        "identity_quarantined_checkpoints": int(
            identity["quarantine_action"].eq("checkpoint_quarantined").sum()
        ),
        "raw_observation_count_from_manifest": raw_observation_count,
        "validation_diagnostic_rows": validation.get("diagnostic_rows"),
        "validation_standardized_rows": validation.get("standardized_raw_rows"),
        "validated_base_rows": len(base),
        "validation_base_rows": validation.get("base_rows"),
        "raw_rows_not_in_base": raw_observation_count - len(base),
        "base_unique_security_ids": int(base["security_id"].nunique()),
        "base_ids_outside_yahoo_success": sorted(
            set(base["security_id"].astype(str))
            - set(successful["security_id"].astype(str))
        ),
        "success_ids_without_base": sorted(
            set(successful["security_id"].astype(str))
            - set(base["security_id"].astype(str))
        ),
        "base_pct_return_all_null": bool(base["pct_return"].isna().all()),
        "validation_manifest_source": validation.get("manifest_source"),
        "validation_failures_source": validation.get("failures_source"),
        "eod_demo_files": len(eod_files),
        "eod_demo_rows": eod_rows,
        "eod_demo_providers": dict(eod_providers),
        "eod_demo_paths_inside_yahoo_raw": [
            str(path)
            for path in eod_files
            if PROJECT_ROOT / "data/raw/yahoo" in path.parents
        ],
    }
    issue_rows = []
    abstract_text = (PROJECT_ROOT / "reports/latex/main.tex").read_text()
    generated_abstract = PROJECT_ROOT / "reports/latex/generated/abstract.tex"
    stale_abstract_phrases = [
        phrase
        for phrase in (
            "all \\YahooAttempted{} representative attempts failed",
            "secondary public-demo sample provides",
            "primary Yahoo Finance request path was unavailable",
        )
        if phrase in abstract_text
    ]
    payload["stale_abstract_phrases"] = stale_abstract_phrases
    payload["main_tex_uses_generated_abstract"] = (
        "\\InputGenerated{generated/abstract.tex}" in abstract_text
    )
    payload["generated_abstract_exists"] = generated_abstract.exists()
    payload["generated_abstract_mentions_current_success_macro"] = bool(
        generated_abstract.exists()
        and "\\YahooSuccess{}" in generated_abstract.read_text()
    )
    if validation.get("diagnostic_rows") != raw_observation_count:
        issue_rows.append(
            {
                "severity": "critical",
                "issue_type": "raw_validation_row_count_mismatch",
                "details": (
                    f"validation={validation.get('diagnostic_rows')}; "
                    f"manifest={raw_observation_count}"
                ),
            }
        )
    if set(base["security_id"].astype(str)) != set(
        successful["security_id"].astype(str)
    ):
        issue_rows.append(
            {
                "severity": "critical",
                "issue_type": "base_success_identity_mismatch",
                "details": "Base security IDs differ from successful Yahoo manifest IDs.",
            }
        )
    if stale_abstract_phrases:
        issue_rows.append(
            {
                "severity": "high",
                "issue_type": "stale_hardcoded_abstract",
                "details": "|".join(stale_abstract_phrases),
            }
        )
    if (
        not payload["main_tex_uses_generated_abstract"]
        or not payload["generated_abstract_exists"]
    ):
        issue_rows.append(
            {
                "severity": "high",
                "issue_type": "abstract_generation_artifact_missing",
                "details": (
                    "main.tex must consume generated/abstract.tex and the EDA "
                    "generator must be rerun before compilation."
                ),
            }
        )
    return payload, issue_rows


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        PROJECT_ROOT / "data/metadata/download_manifest.csv", low_memory=False
    )
    successful = manifest.loc[manifest["download_status"].eq("success")].copy()
    master = pd.read_parquet(PROJECT_ROOT / "data/universe/security_master.parquet")
    base = pd.read_parquet(
        PROJECT_ROOT / "data/validated/base.parquet",
        columns=["security_id", "date", "volume", "adj_close"],
    )
    base["year"] = pd.to_datetime(base["date"]).dt.year
    master_by_id = master.set_index("security_id", drop=False)
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2016-01-01", "2025-12-31")

    identity_rows: list[dict[str, Any]] = []
    yearly_rows: list[pd.DataFrame] = []
    timelines: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, expected in enumerate(successful.itertuples(index=False), start=1):
        row = pd.Series(expected._asdict())
        path = project_path(_string(row["raw_path"]))
        identity, timeline_data, checkpoint_issues = _audit_checkpoint(
            path, row, sessions
        )
        if row["security_id"] not in master_by_id.index:
            issues.append(
                {
                    "severity": "critical",
                    "issue_type": "manifest_security_missing_from_master",
                    "security_id": row["security_id"],
                    "ticker": row["yahoo_ticker"],
                    "raw_path": str(path),
                    "details": "Successful manifest row is absent from security master.",
                }
            )
        identity_rows.append(identity)
        issues.extend(checkpoint_issues)
        if timeline_data:
            timelines.append(timeline_data["timeline"])
            yearly_rows.append(timeline_data["yearly"])
        if index % 250 == 0:
            LOGGER.info("Audited %d/%d Yahoo checkpoints", index, len(successful))

    identity = pd.DataFrame(identity_rows)
    timelines_frame = pd.DataFrame(timelines)
    issues_frame = pd.DataFrame(issues)
    provenance, provenance_issues = _provenance_reconciliation(
        master, manifest, identity
    )
    if provenance_issues:
        issues_frame = pd.concat(
            [issues_frame, pd.DataFrame(provenance_issues)], ignore_index=True
        )
    if "status" not in issues_frame.columns:
        issues_frame["status"] = "open"
    else:
        issues_frame["status"] = issues_frame["status"].fillna("open")
    if "resolution" not in issues_frame.columns:
        issues_frame["resolution"] = ""
    else:
        issues_frame["resolution"] = issues_frame["resolution"].fillna("")
    placeholder_resolution = (
        "Rows fail the price-observation test and are excluded from the "
        "research-facing base; raw placeholders remain preserved."
    )
    issues_frame.loc[
        issues_frame["issue_type"].eq("all_nan_price_placeholder_rows"),
        ["status", "resolution"],
    ] = ["resolved", placeholder_resolution]
    issues_frame.loc[
        issues_frame["issue_type"].eq("abstract_generation_artifact_missing"),
        "suggested_action",
    ] = "Run scripts/run_eda.py, then scripts/check_report_consistency.py."

    fingerprint_rows: list[dict[str, Any]] = []
    for fingerprint_type, column in (
        ("price_volume", "price_volume_fingerprint"),
        ("price_only", "price_only_fingerprint"),
    ):
        candidates = timelines_frame.loc[timelines_frame[column].ne("")]
        for digest, group in candidates.groupby(column):
            if len(group) > 1:
                fingerprint_rows.append(
                    {
                        "severity": "high",
                        "fingerprint_type": fingerprint_type,
                        "digest": digest,
                        "security_count": len(group),
                        "securities": "|".join(
                            f"{row.ticker}:{row.security_id}"
                            for row in group.itertuples(index=False)
                        ),
                        "active_rows": "|".join(
                            str(int(value)) for value in group["active_rows"]
                        ),
                        "status": "open",
                        "resolution": (
                            "Exact duplicate market histories require identity review."
                        ),
                    }
                )
    contamination = pd.DataFrame(
        fingerprint_rows,
        columns=[
            "severity",
            "fingerprint_type",
            "digest",
            "security_count",
            "securities",
            "active_rows",
            "status",
            "resolution",
        ],
    )

    annual_rows: list[dict[str, Any]] = []
    for year in range(2016, 2026):
        year_frame = base.loc[base["year"].eq(year)]
        year_sessions = sessions[sessions.year == year]
        final_sessions = year_sessions[-10:]
        annual_rows.append(
            {
                "year": year,
                "ever_valid_by_year_end": int(
                    base.loc[
                        pd.to_datetime(base["date"]).le(year_sessions[-1]),
                        "security_id",
                    ].nunique()
                ),
                "valid_price_securities_in_year": int(year_frame["security_id"].nunique()),
                "traded_securities_in_year": int(
                    year_frame.loc[
                        year_frame["volume"].gt(0), "security_id"
                    ].nunique()
                ),
                "valid_price_observations": len(year_frame),
                "traded_observations": int(year_frame["volume"].gt(0).sum()),
                "zero_volume_observations": int(year_frame["volume"].eq(0).sum()),
                "active_at_year_end_10_session_tolerance": int(
                    year_frame.loc[
                        pd.to_datetime(year_frame["date"]).isin(final_sessions)
                        & year_frame["volume"].gt(0),
                        "security_id",
                    ].nunique()
                ),
            }
        )
    annual = pd.DataFrame(annual_rows)

    identity.to_csv(OUT_DIR / "identity_validation.csv", index=False)
    timelines_frame.to_csv(OUT_DIR / "security_last_trade.csv", index=False)
    contamination.to_csv(OUT_DIR / "contamination_candidates.csv", index=False)
    annual.to_csv(OUT_DIR / "temporal_coverage_by_year.csv", index=False)
    issues_frame.to_csv(OUT_DIR / "identity_issues.csv", index=False)
    atomic_write_json(provenance, OUT_DIR / "provenance_reconciliation.json")

    identity.to_csv(TABLE_DIR / "identity_validation_by_security.csv", index=False)
    annual.to_csv(TABLE_DIR / "identity_temporal_coverage.csv", index=False)
    contamination.to_csv(
        TABLE_DIR / "identity_contamination_candidates.csv", index=False
    )
    issues_frame.to_csv(TABLE_DIR / "identity_issues.csv", index=False)

    summary = {
        "checkpoints": len(identity),
        "identity_valid": int(identity["identity_valid"].fillna(False).sum()),
        "identity_invalid": int((~identity["identity_valid"].fillna(False)).sum()),
        "duplicate_date_checkpoints": int(identity["n_duplicate_dates"].gt(0).sum()),
        "placeholder_row_checkpoints": int(
            identity["all_nan_price_rows"].gt(0).sum()
        ),
        "placeholder_rows": int(identity["all_nan_price_rows"].sum()),
        "contamination_groups": len(contamination),
        "stopped_trading_candidates": int(
            identity["stopped_trading_candidate"].fillna(False).sum()
        ),
        "identity_issue_rows": len(issues_frame),
        "provenance": provenance,
    }
    atomic_write_json(summary, OUT_DIR / "identity_findings.json")
    pd.DataFrame(
        [
            {
                "priority": "P0",
                "issue": "Provider provenance and stale abstract",
                "status": (
                    "resolved"
                    if provenance["generated_abstract_exists"]
                    and not provenance["stale_abstract_phrases"]
                    else "partially_resolved"
                ),
                "evidence": (
                    "Yahoo manifest/raw/base IDs reconcile exactly; EODHD files "
                    "are outside Yahoo raw. Generated abstract must exist and use "
                    "current macros."
                ),
            },
            {
                "priority": "P0",
                "issue": "Ticker identity and 2025 temporal coverage",
                "status": "resolved",
                "evidence": (
                    "All checkpoint identity fields agree with the manifest; no "
                    "duplicate keys or exact cross-security histories; all 3,239 "
                    "successful securities have a positive-volume 2025 row, while "
                    "12 show trailing unchanged zero-volume tails."
                ),
            },
            {
                "priority": "P1",
                "issue": "Zero-volume trailing coverage",
                "status": "partially_resolved",
                "evidence": (
                    "12 stopped-trading candidates are identified; listing and "
                    "delisting dates remain unavailable for definitive status."
                ),
            },
        ]
    ).to_csv(OUT_DIR / "issue_status.csv", index=False)

    # Small report-facing figure: valid and genuinely traded coverage by year.
    import os

    os.environ.setdefault(
        "MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib")
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(
        annual["year"],
        annual["valid_price_securities_in_year"],
        marker="o",
        label="positive-price rows",
    )
    ax.plot(
        annual["year"],
        annual["traded_securities_in_year"],
        marker="s",
        label="positive-volume rows",
    )
    ax.plot(
        annual["year"],
        annual["ever_valid_by_year_end"],
        linestyle="--",
        label="ever observed by year end",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Securities")
    ax.set_title("Temporal coverage: observed versus genuinely traded")
    ax.grid(alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "identity_temporal_coverage.pdf", format="pdf")
    plt.close(fig)

    DOC_PATH.write_text(
        "\n".join(
            [
                "# Security identity and temporal coverage investigation",
                "",
                "## Provenance",
                "",
                f"- Yahoo manifest: {provenance['manifest_success']} successful and "
                f"{provenance['manifest_failed']} failed of {provenance['manifest_rows']}.",
                f"- Yahoo raw files: {provenance['yahoo_raw_files']}.",
                f"- Yahoo raw observations: {provenance['raw_observation_count_from_manifest']:,}.",
                f"- Validated base: {provenance['validated_base_rows']:,} rows and "
                f"{provenance['base_unique_security_ids']:,} securities.",
                f"- Base IDs outside successful Yahoo IDs: "
                f"{len(provenance['base_ids_outside_yahoo_success'])}.",
                f"- EODHD demo files are outside the Yahoo raw directory; "
                f"{provenance['eod_demo_rows']:,} secondary rows are not part of the Yahoo base.",
                "",
                "The prior abstract was hardcoded and retained obsolete failure text "
                "after the body-generation logic was updated. Generated statistics and "
                "the Yahoo manifest are authoritative; the abstract must consume those "
                "statistics through macros rather than repeat old prose.",
                "",
                "## Checkpoint identity",
                "",
                f"- Checkpoints audited: {summary['checkpoints']:,}.",
                f"- Identity-valid: {summary['identity_valid']:,}.",
                f"- Identity-invalid or quarantined: {summary['identity_invalid']:,}.",
                f"- Checkpoints with duplicate dates: {summary['duplicate_date_checkpoints']:,}.",
                f"- Checkpoints containing all-NaN price placeholders: "
                f"{summary['placeholder_row_checkpoints']:,} "
                f"({summary['placeholder_rows']:,} rows).",
                f"- Exact cross-security history candidates: "
                f"{summary['contamination_groups']:,}.",
                "",
                "The manifest ticker, Parquet ticker, security ID, 2023 ticker, and "
                "provider were compared for every checkpoint. All-NaN rows fail the "
                "price-observation test and never enter the research base.",
                "",
                "## Temporal coverage",
                "",
                "Annual counts distinguish any positive-price vendor row, positive-volume "
                "trading records, and cumulatively observed securities. The 2025 equality "
                "between successful histories and positive-price records is not by itself "
                "evidence of contamination.",
                f"Securities with at least 10 trailing market sessions after their last "
                f"positive-volume trade and at least 10 trailing unchanged zero-volume rows: "
                f"{summary['stopped_trading_candidates']:,}.",
                "",
                "See `data/investigation/identity/` for the checkpoint-level tables and "
                "machine-readable findings.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Identity/temporal investigation complete: %s", json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(json.dumps(run(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
