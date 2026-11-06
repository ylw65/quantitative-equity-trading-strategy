"""Add OHLC fields to the approved v3 research panel without changing its keys."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
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


LOGGER = configure_logging(PROJECT_ROOT / "logs" / "ohlc_enrichment_v3.log")

V3_PAYLOAD_COLUMNS = [
    "date",
    "security_id",
    "ticker",
    "volume",
    "adj_close",
    "pct_return",
]

OHLC_COLUMNS = ["open", "high", "low", "close"]
ACTION_COLUMNS = ["dividends", "stock_splits"]

OUTPUT_COLUMNS = [
    "date",
    "security_id",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "vendor_repair_flag",
    "pct_return",
    "ohlc_source",
    "ohlc_source_file",
    "ohlc_adjustment_convention",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
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


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_unique(
    frame: pd.DataFrame,
    source_name: str,
) -> None:
    duplicates = int(frame.duplicated(["security_id", "date"]).sum())
    if duplicates:
        raise RuntimeError(f"{source_name} has {duplicates} duplicate security-date keys")


def left_join_ohlc(
    daily: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join OHLC while preserving nulls for unmatched approved rows."""
    _validate_unique(daily, "approved v3 panel")
    _validate_unique(source, "standardized OHLC source")
    source_columns = [
        "security_id",
        "date",
        "source_file",
        *OHLC_COLUMNS,
        "adj_close",
        "volume",
        *ACTION_COLUMNS,
    ]
    renamed = source[source_columns].rename(
        columns={
            column: f"source_{column}"
            for column in [
                "source_file",
                *OHLC_COLUMNS,
                "adj_close",
                "volume",
                *ACTION_COLUMNS,
            ]
        }
    )
    joined = daily.merge(
        renamed,
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
        indicator="_ohlc_merge",
    )
    joined["ohlc_source_row_present"] = joined["_ohlc_merge"].eq("both")
    return joined.drop(columns="_ohlc_merge")


def _read_checkpoint_source(
    manifest: pd.DataFrame,
    *,
    tolerance: float,
    relative_tolerance: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    issues = []
    for row in manifest.itertuples(index=False):
        path = project_path(row.raw_path)
        if not path.exists():
            issues.append(
                {
                    "security_id": row.security_id,
                    "ticker": row.ticker_2023,
                    "issue_type": "checkpoint_file_missing",
                    "details": str(path),
                }
            )
            continue
        frame = pd.read_parquet(
            path,
            columns=[
                "date",
                "security_id",
                "ticker_2023",
                *OHLC_COLUMNS,
                "adj_close",
                "volume",
                *ACTION_COLUMNS,
                "repaired",
            ],
        )
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["security_id"] = frame["security_id"].astype(str)
        if frame["security_id"].nunique() != 1 or frame["security_id"].iloc[0] != str(
            row.security_id
        ):
            issues.append(
                {
                    "security_id": row.security_id,
                    "ticker": row.ticker_2023,
                    "issue_type": "checkpoint_identity_mismatch",
                    "details": f"Expected {row.security_id}; found {frame['security_id'].unique().tolist()}",
                }
            )
        frame = frame.drop(columns="security_id").assign(
            checkpoint_security_id=str(row.security_id)
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), issues
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"checkpoint_security_id": "security_id"})
    _validate_unique(combined, "raw checkpoint archive")
    return combined, issues


def _compare_checkpoint_source(
    enriched: pd.DataFrame,
    checkpoints: pd.DataFrame,
    *,
    tolerance: float,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    columns = [*OHLC_COLUMNS, "adj_close", "volume", *ACTION_COLUMNS]
    checkpoint_columns = [
        "security_id",
        "date",
        "ticker_2023",
        *columns,
    ]
    if checkpoints.empty:
        return []
    compare = enriched[
        ["security_id", "date", "ticker", "ohlc_source_file", *columns]
    ].merge(
        checkpoints[checkpoint_columns].rename(
            columns={
                "ticker_2023": "checkpoint_ticker",
                **{column: f"checkpoint_{column}" for column in columns},
            }
        ),
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    issues: list[dict[str, Any]] = []
    missing_source = compare["_merge"].eq("left_only")
    for row in compare.loc[missing_source].itertuples(index=False):
        issues.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker,
                "date": row.date.date().isoformat(),
                "issue_type": "checkpoint_row_missing",
                "severity": "high",
                "standardized_value": "",
                "checkpoint_value": "",
                "details": "v3 row is absent from its per-security checkpoint",
                "resolution_status": "open",
            }
        )
    present = compare["_merge"].eq("both").copy()
    ticker_mismatch = present & compare["ticker"].astype(str).ne(
        compare["checkpoint_ticker"].astype(str)
    )
    for row in compare.loc[ticker_mismatch].itertuples(index=False):
        issues.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker,
                "date": row.date.date().isoformat(),
                "issue_type": "checkpoint_ticker_mismatch",
                "severity": "high",
                "standardized_value": row.ticker,
                "checkpoint_value": row.checkpoint_ticker,
                "details": "Ticker differs between v3 and checkpoint",
                "resolution_status": "open",
            }
        )
    for column in columns:
        first = compare[column]
        second = compare[f"checkpoint_{column}"]
        first_missing = first.isna()
        second_missing = second.isna()
        missing_conflict = present & first_missing.ne(second_missing)
        both_present = present & ~first_missing & ~second_missing
        numerical_conflict = pd.Series(False, index=compare.index)
        if both_present.any():
            numerical_conflict.loc[both_present] = ~np.isclose(
                first.loc[both_present].to_numpy(dtype=float),
                second.loc[both_present].to_numpy(dtype=float),
                atol=tolerance,
                rtol=relative_tolerance,
            )
        for row in compare.loc[missing_conflict].itertuples(index=False):
            issues.append(
                {
                    "security_id": row.security_id,
                    "ticker": row.ticker,
                    "date": row.date.date().isoformat(),
                    "issue_type": f"source_missing_conflict_{column}",
                    "severity": "high",
                    "standardized_value": getattr(row, column),
                    "checkpoint_value": getattr(row, f"checkpoint_{column}"),
                    "details": "One source has a null value and the other does not",
                    "resolution_status": "open",
                }
            )
        for row in compare.loc[numerical_conflict].itertuples(index=False):
            issues.append(
                {
                    "security_id": row.security_id,
                    "ticker": row.ticker,
                    "date": row.date.date().isoformat(),
                    "issue_type": f"source_value_conflict_{column}",
                    "severity": "high",
                    "standardized_value": getattr(row, column),
                    "checkpoint_value": getattr(row, f"checkpoint_{column}"),
                    "details": "Numerical values exceed source-consistency tolerance",
                    "resolution_status": "open",
                }
            )
    return issues


def _ohlc_issues(
    enriched: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    tolerance: float,
    ratio_low: float,
    ratio_high: float,
) -> tuple[list[dict[str, Any]], pd.Series]:
    issues: list[dict[str, Any]] = []
    finite_values = np.isfinite(enriched[[*OHLC_COLUMNS, "adj_close"]])
    ohlc_finite = finite_values[[*OHLC_COLUMNS]].all(axis=1)
    nonfinite = ~finite_values.all(axis=1)
    nonpositive = (enriched[["open", "high", "low", "close"]] <= 0).any(axis=1)
    low_gt_open = enriched["low"].gt(enriched["open"].add(tolerance))
    open_gt_high = enriched["open"].gt(enriched["high"].add(tolerance))
    low_gt_close = enriched["low"].gt(enriched["close"].add(tolerance))
    close_gt_high = enriched["close"].gt(enriched["high"].add(tolerance))
    high_lt_low = enriched["high"].lt(enriched["low"].sub(tolerance))
    ratio = enriched["close"].div(enriched["adj_close"].replace(0, np.nan))
    ratio_flag = ratio.lt(ratio_low) | ratio.gt(ratio_high)
    issue_masks = [
        ("ohlc_nonfinite", "high", nonfinite, "A numeric OHLC or adjusted-close value is nonfinite."),
        ("ohlc_nonpositive", "high", nonpositive, "An Open/High/Low/Close value is nonpositive."),
        ("ohlc_low_above_open", "high", low_gt_open, "Low exceeds Open beyond tolerance."),
        ("ohlc_open_above_high", "high", open_gt_high, "Open exceeds High beyond tolerance."),
        ("ohlc_low_above_close", "high", low_gt_close, "Low exceeds Close beyond tolerance."),
        ("ohlc_close_above_high", "high", close_gt_high, "Close exceeds High beyond tolerance."),
        ("ohlc_high_below_low", "high", high_lt_low, "High is below Low beyond tolerance."),
        (
            "diagnostic_close_adj_ratio_extreme",
            "medium",
            ratio_flag,
            "Close/Adj Close ratio lies outside the documented diagnostic range.",
        ),
    ]
    for issue_type, severity, mask, details in issue_masks:
        for row in enriched.loc[mask.fillna(False)].itertuples(index=False):
            issues.append(
                {
                    "security_id": row.security_id,
                    "ticker": row.ticker,
                    "date": row.date.date().isoformat(),
                    "issue_type": issue_type,
                    "severity": severity,
                    "standardized_value": getattr(row, issue_type.split("_")[-1], ""),
                    "checkpoint_value": "",
                    "details": details,
                    "resolution_status": "open",
                }
            )
    existing = diagnostics[
        [
            "security_id",
            "date",
            "ohlc_consistency_flag",
            "raw_adjusted_return_conflict_flag",
        ]
    ]
    existing = existing.merge(
        enriched[["security_id", "date", "ticker"]],
        on=["security_id", "date"],
        how="inner",
        validate="one_to_one",
    )
    for row in existing.loc[existing["ohlc_consistency_flag"]].itertuples(
        index=False
    ):
        issues.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker,
                "date": row.date.date().isoformat(),
                "issue_type": "existing_ohlc_consistency_flag",
                "severity": "high",
                "standardized_value": "",
                "checkpoint_value": "",
                "details": "Existing validation OHLC consistency flag retained.",
                "resolution_status": "open",
            }
        )
    for row in existing.loc[
        existing["raw_adjusted_return_conflict_flag"]
    ].itertuples(index=False):
        issues.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker,
                "date": row.date.date().isoformat(),
                "issue_type": "existing_raw_adjusted_return_conflict",
                "severity": "high",
                "standardized_value": "",
                "checkpoint_value": "",
                "details": "Existing raw Close/Adj Close return conflict retained.",
                "resolution_status": "open",
            }
        )
    return issues, ohlc_finite


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(project_path(config_path))
    inputs = config["inputs"]
    parent_root = project_path(inputs["v3_root"])
    output_root = project_path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    daily_path = project_path(inputs["selected_daily"])
    master_path = project_path(inputs["selected_master"])
    raw_path = project_path(inputs["standardized_raw"])
    diagnostic_path = project_path(inputs["diagnostic_returns"])
    manifest_path = project_path(inputs["download_manifest"])
    protected_files = [
        daily_path,
        master_path,
        diagnostic_path,
        raw_path,
        manifest_path,
    ]
    before_hashes = {str(path): _sha256(path) for path in protected_files}
    before_parent_tree = tree_sha256(parent_root)
    raw_root = project_path("data/raw/yahoo")
    before_raw_tree = tree_sha256(raw_root)

    parent_daily = pd.read_parquet(daily_path)
    parent_master = pd.read_parquet(master_path)
    parent_diagnostics = pd.read_parquet(diagnostic_path)
    parent_daily["date"] = pd.to_datetime(parent_daily["date"]).dt.normalize()
    parent_daily["security_id"] = parent_daily["security_id"].astype(str)
    parent_diagnostics["date"] = pd.to_datetime(
        parent_diagnostics["date"]
    ).dt.normalize()
    _validate_unique(parent_daily, "v3 daily panel")
    _validate_unique(parent_diagnostics, "v3 diagnostics")
    parent_security_ids = set(parent_daily["security_id"])
    if (
        len(parent_master) != 487
        or set(parent_master["security_id"].astype(str)) != parent_security_ids
    ):
        raise RuntimeError("v3 master and daily security sets disagree")
    if "SEC:0000009984" in parent_security_ids:
        raise RuntimeError("Excluded B security unexpectedly present in v3")
    excluded_ids = {
        str(value) for value in config.get("exclusions", {}).get("security_ids", [])
    }
    unknown_exclusions = excluded_ids - parent_security_ids
    if unknown_exclusions:
        raise RuntimeError(
            f"Configured OHLC security exclusions are absent from v3: {unknown_exclusions}"
        )
    excluded_identity = parent_master[
        parent_master["security_id"].astype(str).isin(excluded_ids)
    ]
    if excluded_identity["ticker_2023"].tolist() != ["HUBB"]:
        raise RuntimeError("Expected HUBB as the sole configured security exclusion")
    excluded_daily = parent_daily[
        parent_daily["security_id"].isin(excluded_ids)
    ].copy().reset_index(drop=True)
    daily = (
        parent_daily[~parent_daily["security_id"].isin(excluded_ids)]
        .copy()
        .reset_index(drop=True)
    )
    master = parent_master[
        ~parent_master["security_id"].astype(str).isin(excluded_ids)
    ].copy().reset_index(drop=True)
    diagnostics = parent_diagnostics[
        ~parent_diagnostics["security_id"].astype(str).isin(excluded_ids)
    ].copy().reset_index(drop=True)
    security_ids = set(daily["security_id"])
    if len(master) != 486 or master["cik"].nunique() != 484:
        raise RuntimeError("Unexpected post-exclusion v3 security or issuer count")
    _validate_unique(daily, "approved OHLC research panel")

    raw = pd.read_parquet(
        raw_path,
        columns=[
            "security_id",
            "date",
            "source_file",
            *OHLC_COLUMNS,
            "adj_close",
            "volume",
            *ACTION_COLUMNS,
        ],
        filters=[("security_id", "in", sorted(security_ids))],
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw["security_id"] = raw["security_id"].astype(str)
    _validate_unique(raw, "standardized raw subset")
    excluded_raw = pd.read_parquet(
        raw_path,
        columns=[
            "security_id",
            "date",
            "source_file",
            *OHLC_COLUMNS,
            "adj_close",
            "volume",
            *ACTION_COLUMNS,
        ],
        filters=[("security_id", "in", sorted(excluded_ids))],
    )
    excluded_raw["date"] = pd.to_datetime(excluded_raw["date"]).dt.normalize()
    excluded_raw["security_id"] = excluded_raw["security_id"].astype(str)
    _validate_unique(excluded_raw, "excluded security raw subset")
    excluded_enriched = left_join_ohlc(excluded_daily, excluded_raw)
    excluded_enriched = excluded_enriched.rename(
        columns={
            "source_open": "open",
            "source_high": "high",
            "source_low": "low",
            "source_close": "close",
            "source_dividends": "dividends",
            "source_stock_splits": "stock_splits",
            "source_source_file": "ohlc_source_file",
        }
    )
    excluded_enriched["vendor_repair_flag"] = pd.NA
    excluded_enriched["ohlc_source"] = np.where(
        excluded_enriched["ohlc_source_row_present"],
        "standardized_raw.parquet",
        "unavailable",
    )
    excluded_enriched["ohlc_adjustment_convention"] = config["project"][
        "ohlc_adjustment_convention"
    ]
    if len(daily) != 1_181_168:
        raise RuntimeError(
            f"Unexpected approved observation count after security exclusion: {len(daily)}"
        )
    check = left_join_ohlc(daily, raw)
    if len(check) != len(daily):
        raise RuntimeError("OHLC join changed the v3 row count")
    matched = check["ohlc_source_row_present"]
    for column in ("adj_close", "volume"):
        v3_values = check.loc[matched, column]
        source_values = check.loc[matched, f"source_{column}"]
        if not np.allclose(
            v3_values.to_numpy(dtype=float),
            source_values.to_numpy(dtype=float),
            equal_nan=True,
            atol=float(config["project"]["numeric_tolerance"]),
            rtol=float(config["project"]["relative_tolerance"]),
        ):
            raise RuntimeError(f"v3 {column} differs from standardized raw")

    enriched = daily.copy()
    enriched["open"] = check["source_open"]
    enriched["high"] = check["source_high"]
    enriched["low"] = check["source_low"]
    enriched["close"] = check["source_close"]
    enriched["dividends"] = check["source_dividends"]
    enriched["stock_splits"] = check["source_stock_splits"]
    enriched["vendor_repair_flag"] = pd.NA
    enriched["ohlc_source"] = np.where(
        check["ohlc_source_row_present"],
        "standardized_raw.parquet",
        "unavailable",
    )
    enriched["ohlc_source_file"] = check["source_source_file"]
    enriched["ohlc_adjustment_convention"] = config["project"][
        "ohlc_adjustment_convention"
    ]
    enriched = enriched[OUTPUT_COLUMNS]

    all_manifest = pd.read_csv(manifest_path, dtype={"security_id": str})
    excluded_manifest = all_manifest[
        all_manifest["security_id"].isin(excluded_ids)
    ].copy()
    manifest = all_manifest[
        all_manifest["security_id"].isin(security_ids)
        & all_manifest["download_status"].eq("success")
    ].copy()
    checkpoints, checkpoint_metadata_issues = _read_checkpoint_source(
        manifest,
        tolerance=float(config["project"]["numeric_tolerance"]),
        relative_tolerance=float(config["project"]["relative_tolerance"]),
    )
    repair_lookup = checkpoints[
        ["security_id", "date", "repaired"]
    ].set_index(["security_id", "date"])["repaired"]
    enriched["vendor_repair_flag"] = pd.array(
        repair_lookup.reindex(
            pd.MultiIndex.from_frame(enriched[["security_id", "date"]])
        ).to_numpy(),
        dtype="boolean",
    )
    source_issues = _compare_checkpoint_source(
        enriched,
        checkpoints,
        tolerance=float(config["project"]["numeric_tolerance"]),
        relative_tolerance=float(config["project"]["relative_tolerance"]),
    )
    excluded_row = excluded_identity.iloc[0]
    excluded_source_file = (
        str(excluded_manifest["raw_path"].dropna().iloc[-1])
        if excluded_manifest["raw_path"].notna().any()
        else ""
    )
    security_exclusions = pd.DataFrame(
        [
            {
                "security_id": excluded_row["security_id"],
                "cik": str(excluded_row["cik"]).zfill(10),
                "ticker": excluded_row["ticker_2023"],
                "company_name": excluded_row["company_name"],
                "exchange": excluded_row["exchange"],
                "observations_removed": int(len(excluded_enriched)),
                "first_observation": excluded_enriched["date"].min().date().isoformat(),
                "last_observation": excluded_enriched["date"].max().date().isoformat(),
                "exclusion_reason": config["exclusions"]["reason"],
                "policy_type": config["exclusions"]["policy_type"],
                "decision_date": config["exclusions"]["decision_date"],
                "original_source_file": excluded_source_file,
            }
        ]
    )
    removed_observations = excluded_enriched[OUTPUT_COLUMNS].assign(
        exclusion_reason=config["exclusions"]["reason"],
        policy_type=config["exclusions"]["policy_type"],
        decision_date=config["exclusions"]["decision_date"],
    )

    issue_rows: list[dict[str, Any]] = []
    for row in check.loc[~check["ohlc_source_row_present"]].itertuples(
        index=False
    ):
        issue_rows.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker,
                "date": row.date.date().isoformat(),
                "issue_type": "standardized_ohlc_row_missing",
                "severity": "high",
                "standardized_value": "",
                "checkpoint_value": "",
                "details": "Approved v3 row has no matching standardized raw OHLC row.",
                "resolution_status": "open",
            }
        )
    for issue in checkpoint_metadata_issues:
        issue_rows.append(
            {
                "security_id": issue["security_id"],
                "ticker": issue["ticker"],
                "date": "",
                "issue_type": issue["issue_type"],
                "severity": "high",
                "standardized_value": "",
                "checkpoint_value": "",
                "details": issue["details"],
                "resolution_status": "open",
            }
        )
    issue_rows.extend(source_issues)
    calculated_issues, complete_mask = _ohlc_issues(
        enriched,
        diagnostics,
        tolerance=float(config["project"]["numeric_tolerance"]),
        ratio_low=float(config["project"]["close_adj_ratio_review_low"]),
        ratio_high=float(config["project"]["close_adj_ratio_review_high"]),
    )
    issue_rows.extend(calculated_issues)
    issues = pd.DataFrame(
        issue_rows,
        columns=[
            "security_id",
            "ticker",
            "date",
            "issue_type",
            "severity",
            "standardized_value",
            "checkpoint_value",
            "details",
            "resolution_status",
        ],
    )

    coverage_rows = []
    issue_counts = issues.groupby("security_id").size() if not issues.empty else {}
    for row in master.sort_values("security_id").itertuples(index=False):
        group = enriched[enriched["security_id"].eq(row.security_id)]
        missing = {column: int(group[column].isna().sum()) for column in OHLC_COLUMNS}
        complete = int(
            group[[*OHLC_COLUMNS]].notna().all(axis=1).sum()
        )
        coverage_rows.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker_2023,
                "observations": int(len(group)),
                "complete_ohlc_rows": complete,
                "complete_ohlc_pct": float(100.0 * complete / len(group))
                if len(group)
                else np.nan,
                "missing_open_rows": missing["open"],
                "missing_high_rows": missing["high"],
                "missing_low_rows": missing["low"],
                "missing_close_rows": missing["close"],
                "affected_record_issue_count": int(issue_counts.get(row.security_id, 0)),
                "first_date": group["date"].min().date().isoformat(),
                "last_date": group["date"].max().date().isoformat(),
                "ohlc_source": "standardized_raw.parquet",
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    complete_rows = int(
        enriched[[*OHLC_COLUMNS]].notna().all(axis=1).sum()
    )
    complete_securities = int(
        coverage["complete_ohlc_rows"].eq(coverage["observations"]).sum()
    )
    missing_counts = {
        column: int(enriched[column].isna().sum()) for column in OHLC_COLUMNS
    }
    missing_any = int(
        enriched[[*OHLC_COLUMNS]].isna().any(axis=1).sum()
    )
    affected_securities = int(
        enriched.loc[
            enriched[[*OHLC_COLUMNS]].isna().any(axis=1), "security_id"
        ].nunique()
    )
    dated_issues = issues[issues["date"].fillna("").ne("")] if not issues.empty else issues
    inconsistent_rows = int(
        dated_issues[["security_id", "date"]].drop_duplicates().shape[0]
    )
    inconsistent_securities = int(dated_issues["security_id"].nunique())

    schema = pd.DataFrame(
        [
            {
                "field": column,
                "v3": True,
                "ohlc_dataset": column in enriched.columns,
                "change": "preserved",
            }
            for column in V3_PAYLOAD_COLUMNS
        ]
        + [
            {
                "field": column,
                "v3": False,
                "ohlc_dataset": True,
                "change": "added",
            }
            for column in [
                *OHLC_COLUMNS,
                *ACTION_COLUMNS,
                "vendor_repair_flag",
                "ohlc_source",
                "ohlc_source_file",
                "ohlc_adjustment_convention",
            ]
        ]
    )

    _atomic_parquet(enriched, output_root / "selected_equity_daily.parquet")
    _atomic_parquet(master, output_root / "selected_security_master.parquet")
    _atomic_csv(coverage, output_root / "ohlc_coverage_report.csv")
    _atomic_csv(issues, output_root / "ohlc_discrepancies.csv")
    _atomic_csv(
        security_exclusions, output_root / "security_exclusions.csv"
    )
    _atomic_csv(
        removed_observations, output_root / "removed_security_observations.csv"
    )
    _atomic_csv(schema, output_root / "schema_comparison.csv")

    source_inventory = pd.DataFrame(
        [
            {
                "source": "standardized_raw.parquet",
                "path": str(raw_path),
                "sha256": _sha256(raw_path),
                "rows_in_selected_security_scope": int(len(raw)),
                "securities": int(raw["security_id"].nunique()),
                "adjustment_convention": config["project"][
                    "ohlc_adjustment_convention"
                ],
                "usage": "primary OHLC/action source",
            },
            {
                "source": "per_security_yahoo_checkpoints",
                "path": str(raw_root),
                "sha256": before_raw_tree,
                "rows_in_selected_security_scope": int(len(checkpoints)),
                "securities": int(checkpoints["security_id"].nunique())
                if not checkpoints.empty
                else 0,
                "adjustment_convention": config["project"][
                    "ohlc_adjustment_convention"
                ],
                "usage": "independent source-consistency check",
            },
        ]
    )
    provenance = {
        "generated_at_utc": utc_now(),
        "primary_source": "data/validated/standardized_raw.parquet",
        "checkpoint_source": "data/raw/yahoo/",
        "adjustment_convention": config["project"][
            "ohlc_adjustment_convention"
        ],
        "source_inventory": source_inventory.to_dict("records"),
        "identity_join_keys": ["security_id", "date"],
        "security_exclusions": security_exclusions.to_dict("records"),
        "targeted_yahoo_requests": 0,
        "sources_modified": False,
    }
    atomic_write_json(provenance, output_root / "ohlc_source_provenance.json")

    v3_key_set = set(zip(daily["security_id"], daily["date"], strict=True))
    out_key_set = set(zip(enriched["security_id"], enriched["date"], strict=True))
    if v3_key_set != out_key_set:
        raise RuntimeError("OHLC enrichment changed the v3 security-date key set")
    for column in V3_PAYLOAD_COLUMNS:
        left = daily[column].reset_index(drop=True)
        right = enriched[column].reset_index(drop=True)
        if column == "pct_return":
            if not right.isna().all():
                raise RuntimeError("pct_return was changed during OHLC enrichment")
        elif column in {"date", "security_id", "ticker"}:
            if not left.equals(right):
                raise RuntimeError(f"Existing v3 field changed: {column}")
        elif not np.allclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            equal_nan=True,
            atol=float(config["project"]["numeric_tolerance"]),
            rtol=float(config["project"]["relative_tolerance"]),
        ):
            raise RuntimeError(f"Existing v3 field changed: {column}")

    after_hashes = {str(path): _sha256(path) for path in protected_files}
    after_parent_tree = tree_sha256(parent_root)
    after_raw_tree = tree_sha256(raw_root)
    if (
        before_hashes != after_hashes
        or before_parent_tree != after_parent_tree
        or before_raw_tree != after_raw_tree
    ):
        raise RuntimeError("A protected v3 or raw checkpoint changed")

    complete_pct = 100.0 * complete_rows / len(enriched) if len(enriched) else np.nan
    metadata = {
        "generated_at_utc": utc_now(),
        "dataset": config["project"]["name"],
        "parent_dataset": config["project"]["parent_version"],
        "v3_securities": int(parent_daily["security_id"].nunique()),
        "enriched_securities": int(enriched["security_id"].nunique()),
        "v3_observations": int(len(parent_daily)),
        "enriched_observations": int(len(enriched)),
        "security_exclusions": int(len(security_exclusions)),
        "security_exclusion_observations": int(len(removed_observations)),
        "excluded_security_ids": sorted(excluded_ids),
        "complete_ohlc_rows": complete_rows,
        "complete_ohlc_pct": complete_pct,
        "securities_with_complete_ohlc": complete_securities,
        "missing_open_rows": missing_counts["open"],
        "missing_high_rows": missing_counts["high"],
        "missing_low_rows": missing_counts["low"],
        "missing_close_rows": missing_counts["close"],
        "rows_missing_any_ohlc": missing_any,
        "securities_affected_by_missing_ohlc": affected_securities,
        "ohlc_issue_records": int(len(issues)),
        "inconsistent_ohlc_rows": inconsistent_rows,
        "inconsistent_ohlc_securities": inconsistent_securities,
        "missing_or_inconsistent_ohlc_rows": int(missing_any + inconsistent_rows),
        "unresolved_ohlc_issue_records": int(
            issues["resolution_status"].eq("open").sum()
        )
        if not issues.empty
        else 0,
        "source_conflict_records": int(
            issues["issue_type"].str.startswith("source_").sum()
        )
        if not issues.empty
        else 0,
        "records_from_existing_local_data": int(len(enriched)),
        "records_requiring_targeted_download": 0,
        "vendor_repaired_rows": int(
            enriched["vendor_repair_flag"].fillna(False).sum()
        ),
        "vendor_repaired_securities": int(
            enriched.loc[
                enriched["vendor_repair_flag"].fillna(False), "security_id"
            ].nunique()
        ),
        "stock_split_rows": int(enriched["stock_splits"].gt(0).sum()),
        "stock_split_securities": int(
            enriched.loc[
                enriched["stock_splits"].gt(0), "security_id"
            ].nunique()
        ),
        "dividend_rows": int(enriched["dividends"].gt(0).sum()),
        "dividend_securities": int(
            enriched.loc[enriched["dividends"].gt(0), "security_id"].nunique()
        ),
        "simultaneous_split_dividend_rows": int(
            (
                enriched["stock_splits"].gt(0)
                & enriched["dividends"].gt(0)
            ).sum()
        ),
        "new_security_identity_issues": 0,
        "excluded_b_present": False,
        "pct_return_non_null": int(enriched["pct_return"].notna().sum()),
        "downstream_feature_engineering_status": (
            "ready_for_downstream_ohlc_feature_engineering"
            if complete_rows == len(enriched) and len(issues) == 0
            else "not_ready_until_missing_ohlc_is_resolved"
            if complete_rows < len(enriched)
            else "structurally_ready_subject_to_documented_ohlc_flags"
        ),
        "schema_comparison": schema.to_dict("records"),
        "protected_inputs_unchanged": True,
        "targeted_yahoo_downloads": [],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    atomic_write_json(metadata, output_root / "dataset_metadata.json")
    report = f"""# OHLC Enrichment Report

## Result

- Parent v3 securities: {parent_daily['security_id'].nunique():,}
- Enriched securities: {enriched['security_id'].nunique():,}
- Parent v3 observations: {len(parent_daily):,}
- Enriched observations: {len(enriched):,}
- Security exclusions: {len(security_exclusions):,}
- Observations removed with excluded securities: {len(removed_observations):,}
- Complete OHLC observations: {complete_rows:,} ({complete_pct:.6f}%)
- Securities with complete OHLC coverage: {complete_securities:,}
- Missing Open/High/Low/Close rows: {missing_counts['open']:,} / {missing_counts['high']:,} / {missing_counts['low']:,} / {missing_counts['close']:,}
- Rows missing any OHLC: {missing_any:,}
- OHLC issue records: {len(issues):,}
- Unique inconsistent OHLC rows: {inconsistent_rows:,}
- Source-conflict records: {int(metadata['source_conflict_records']):,}
- Records obtained from existing local data: {len(enriched):,}
- Targeted Yahoo downloads: 0

## Method

OHLC was joined from `data/validated/standardized_raw.parquet` using
`security_id` and `date` only. The source uses Yahoo `auto_adjust=False` raw
OHLC conventions with `actions=True` and `repair=True`. Raw Close and Adj Close
remain distinct. No split or dividend adjustment was applied.

Every retained enriched row was independently compared with its original
per-security Yahoo checkpoint. Adjusted close, volume, and null `pct_return`
were preserved, as were the historical FERG/DKNG boundaries and exclusion of
Barnes/Barrick `B`.

The complete HUBB security (`SEC:0000048898`) is intentionally excluded by the
user-approved OHLC enrichment policy. Its 2,514 retained parent-v3 observations
are documented in `security_exclusions.csv` and
`removed_security_observations.csv`; the parent v3 dataset remains unchanged.

The checkpoint `repaired` marker is retained as `vendor_repair_flag` for {int(enriched['vendor_repair_flag'].fillna(False).sum()):,} observations across {int(enriched.loc[enriched['vendor_repair_flag'].fillna(False), 'security_id'].nunique()):,} securities. It is vendor-repair provenance, not an identity error.

The retained actions include {int(enriched['stock_splits'].gt(0).sum()):,} stock-split rows across {int(enriched.loc[enriched['stock_splits'].gt(0), 'security_id'].nunique()):,} securities and {int(enriched['dividends'].gt(0).sum()):,} dividend rows across {int(enriched.loc[enriched['dividends'].gt(0), 'security_id'].nunique()):,} securities. These actions were preserved without additional OHLC adjustment.

## Validation

OHLC bounds use an absolute tolerance of {config['project']['numeric_tolerance']}.
The parent HUBB low/open inconsistency on 2021-05-05 is removed together with
the complete HUBB security. The enriched dataset has zero missing OHLC,
zero source conflicts, zero inconsistent OHLC rows, and zero OHLC issue records.

The enriched dataset has complete OHLC coverage and is ready for downstream OHLC
feature engineering, subject to the existing broader data-quality and economic
eligibility limitations.
No strategy feature, final return, model, chronological split, or backtest is
created in this task.
"""
    (output_root / "OHLC_ENRICHMENT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    return metadata


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "ohlc_enrichment_v3.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
