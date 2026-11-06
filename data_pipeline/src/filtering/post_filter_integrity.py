"""Assess missingness, identity, and data quality after ADTV filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import exchange_calendars as xcals

from src.common import (
    PROJECT_ROOT,
    atomic_write_json,
    configure_logging,
    load_config,
    project_path,
)


COLORS = {
    "navy": "#19324D",
    "blue": "#3F6F8F",
    "gold": "#B28A48",
    "rust": "#A45D4F",
    "slate": "#65727A",
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_tex(frame: pd.DataFrame, path: Path, align: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = align or ("l" + "r" * max(0, len(frame.columns) - 1))
    lines = [
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(str(column).replace("_", r"\_") for column in frame.columns)
        + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (int, np.integer)):
                cells.append(f"{int(value):,}")
            elif isinstance(value, (float, np.floating)):
                cells.append("Unknown" if not np.isfinite(value) else f"{value:,.3f}")
            else:
                cells.append(
                    "Unknown"
                    if value is None or pd.isna(value)
                    else str(value).replace("_", r"\_").replace("&", r"\&")
                )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_intervals(
    frame: pd.DataFrame,
    value_column: str,
    *,
    security_id: str,
    ticker: str,
) -> list[dict[str, Any]]:
    """Describe contiguous missing-value intervals within retained raw rows."""
    if frame.empty:
        return []
    ordered = frame.sort_values("date", kind="stable").reset_index(drop=True)
    missing = ordered[value_column].isna()
    if not missing.any():
        return []
    valid_positions = np.flatnonzero(ordered[value_column].notna().to_numpy())
    first_valid = int(valid_positions.min()) if len(valid_positions) else len(ordered)
    last_valid = int(valid_positions.max()) if len(valid_positions) else -1
    transitions = np.diff(np.r_[False, missing.to_numpy(), False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    rows = []
    for start, end in zip(starts, ends, strict=True):
        if end < first_valid:
            interval_class = "leading_unknown_placeholder_or_pre_history"
        elif start > last_valid:
            interval_class = "trailing_unknown_placeholder_or_post_history"
        else:
            interval_class = "internal"
        rows.append(
            {
                "security_id": security_id,
                "ticker": ticker,
                "field": value_column,
                "start_date": pd.Timestamp(ordered.at[start, "date"]).date().isoformat(),
                "end_date": pd.Timestamp(ordered.at[end, "date"]).date().isoformat(),
                "row_count": int(end - start + 1),
                "interval_class": interval_class,
                "cause": "unknown_from_provider_fields",
            }
        )
    return rows


def calendar_gap_intervals(
    observed: pd.DatetimeIndex,
    calendar_sessions: pd.DatetimeIndex,
    *,
    security_id: str,
    ticker: str,
) -> list[dict[str, Any]]:
    """Describe internal exchange sessions absent between observed endpoints."""
    if len(observed) == 0:
        return []
    first = observed.min()
    last = observed.max()
    expected = calendar_sessions[
        (calendar_sessions >= first) & (calendar_sessions <= last)
    ]
    missing = expected.difference(observed)
    if len(missing) == 0:
        return []
    indexes = np.searchsorted(calendar_sessions, missing)
    breaks = np.flatnonzero(np.diff(indexes) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(indexes) - 1]
    return [
        {
            "security_id": security_id,
            "ticker": ticker,
            "start_date": missing[start].date().isoformat(),
            "end_date": missing[end].date().isoformat(),
            "session_count": int(end - start + 1),
            "interval_class": "internal_calendar_gap",
            "cause": "unknown_without_listing_or_suspension_evidence",
        }
        for start, end in zip(starts, ends, strict=True)
    ]


def identity_review_candidates(
    identity: pd.DataFrame, selected_ids: set[str]
) -> pd.DataFrame:
    """Return selected identities carrying a non-``none`` review action."""
    action = identity["quarantine_action"].fillna("none").astype(str)
    return identity.loc[
        identity["security_id"].astype(str).isin(selected_ids)
        & action.ne("none")
    ].copy()


def _selection_coverage(
    selected_master: pd.DataFrame,
    selected_daily: pd.DataFrame,
    calendar_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    dates_by_id = {
        security_id: pd.DatetimeIndex(group["date"].unique())
        for security_id, group in selected_daily.groupby("security_id", sort=False)
    }
    rows = []
    start = calendar_sessions[0]
    end = calendar_sessions[-1]
    for row in selected_master.itertuples(index=False):
        observed = dates_by_id.get(row.security_id, pd.DatetimeIndex([]))
        if len(observed):
            first = observed.min()
            last = observed.max()
            leading = int((calendar_sessions < first).sum())
            trailing = int((calendar_sessions > last).sum())
            internal = int(
                len(
                    calendar_sessions[
                        (calendar_sessions >= first) & (calendar_sessions <= last)
                    ].difference(observed)
                )
            )
        else:
            first = pd.NaT
            last = pd.NaT
            leading = trailing = internal = int(len(calendar_sessions))
        rows.append(
            {
                "security_id": row.security_id,
                "ticker": row.ticker_2023,
                "first_observation": first,
                "last_observation": last,
                "leading_sessions_unknown": leading,
                "internal_calendar_gap_sessions": internal,
                "trailing_sessions_unknown": trailing,
                "incomplete_2016_2025_interval": bool(leading or trailing),
                "structural_classification": (
                    "unknown_no_verified_listing_or_delisting_dates"
                ),
                "calendar_start": start,
                "calendar_end": end,
            }
        )
    return pd.DataFrame(rows)


def _raw_metrics(
    raw: pd.DataFrame,
    base: pd.DataFrame,
    ids: set[str],
) -> dict[str, int]:
    scoped = raw[raw["security_id"].astype(str).isin(ids)]
    b = base[base["security_id"].astype(str).isin(ids)]
    raw_keys = set(zip(scoped["security_id"], scoped["date"], strict=True))
    base_keys = set(zip(b["security_id"], b["date"], strict=True))
    all_nan = scoped[
        ["open", "high", "low", "close", "adj_close"]
    ].isna().all(axis=1)
    return {
        "raw_rows": int(len(scoped)),
        "validated_rows": int(len(b)),
        "raw_rows_not_in_base": int(len(raw_keys - base_keys)),
        "placeholder_all_nan_rows": int(all_nan.sum()),
        "missing_adj_close_rows": int(scoped["adj_close"].isna().sum()),
        "missing_volume_rows": int(scoped["volume"].isna().sum()),
        "nonpositive_adj_close_rows": int(scoped["adj_close"].le(0).sum()),
        "negative_volume_rows": int(scoped["volume"].lt(0).sum()),
        "zero_volume_rows": int(scoped["volume"].eq(0).sum()),
        "duplicate_security_date_rows": int(
            scoped.duplicated(["security_id", "date"]).sum()
        ),
    }


def _before_after(
    config: dict[str, Any],
    selected_ids: set[str],
    selected_master: pd.DataFrame,
    selected_base: pd.DataFrame,
    selected_raw: pd.DataFrame,
    selected_diagnostics: pd.DataFrame,
    selected_quality: pd.DataFrame,
    quality_all: pd.DataFrame,
    reconciliation: pd.DataFrame,
    quarantine: pd.DataFrame,
) -> pd.DataFrame:
    acquired = quality_all[quality_all["download_status"].eq("success")]
    acquired_ids = set(acquired["security_id"].astype(str))
    master = pd.read_parquet(project_path(config["inputs"]["security_master"]))
    acquired_master = master[master["security_id"].astype(str).isin(acquired_ids)]
    selected_reconciliation = reconciliation[
        reconciliation["security_id"].astype(str).isin(selected_ids)
    ]
    selected_quarantine = quarantine[
        quarantine["security_id"].astype(str).isin(selected_ids)
    ]
    all_extremes = pd.read_csv(
        PROJECT_ROOT / "data/investigation/anomalies/extreme_return_cases.csv",
        low_memory=False,
    )
    all_unresolved = all_extremes[
        all_extremes["classification"].eq("unresolved_extreme_discontinuity")
    ]
    selected_extremes = all_extremes[
        all_extremes["security_id"].astype(str).isin(selected_ids)
    ]
    selected_unresolved = selected_extremes[
        selected_extremes["classification"].eq("unresolved_extreme_discontinuity")
    ]

    base_all = pd.read_parquet(project_path(config["inputs"]["base"]))
    original_raw_rows = int(acquired["trading_sessions"].sum())
    filtered_raw_rows = len(selected_raw)
    original_events = int(
        acquired["split_events"].sum() + acquired["dividend_events"].sum()
    )
    filtered_events = int(
        selected_quality["split_events"].sum()
        + selected_quality["dividend_events"].sum()
    )

    metric_specs = [
        ("Securities", len(acquired), len(selected_master), None, None, len(acquired_ids), len(selected_ids)),
        (
            "Distinct SEC issuers",
            acquired_master["cik"].nunique(),
            selected_master["cik"].nunique(),
            None,
            None,
            acquired_master["cik"].nunique(),
            selected_master["cik"].nunique(),
        ),
        ("Validated stock-day rows", len(base_all), len(selected_base), original_raw_rows, filtered_raw_rows, len(acquired_ids), len(selected_ids)),
        (
            "Missing adjusted-close rows",
            int(acquired["missing_adj_close_observations"].sum()),
            int(selected_quality["missing_adj_close_observations"].sum()),
            original_raw_rows,
            filtered_raw_rows,
            int(acquired["missing_adj_close_observations"].gt(0).sum()),
            int(selected_quality["missing_adj_close_observations"].gt(0).sum()),
        ),
        (
            "Missing-volume rows",
            int(acquired["missing_volume_observations"].sum()),
            int(selected_quality["missing_volume_observations"].sum()),
            original_raw_rows,
            filtered_raw_rows,
            int(acquired["missing_volume_observations"].gt(0).sum()),
            int(selected_quality["missing_volume_observations"].gt(0).sum()),
        ),
        (
            "Internal calendar gaps",
            int(acquired["unobserved_internal_sessions"].sum()),
            int(selected_quality["unobserved_internal_sessions"].sum()),
            int(acquired["expected_sessions_observed_interval"].sum()),
            int(selected_quality["expected_sessions_observed_interval"].sum()),
            int(acquired["unobserved_internal_sessions"].gt(0).sum()),
            int(selected_quality["unobserved_internal_sessions"].gt(0).sum()),
        ),
        (
            "Zero-volume rows",
            int(acquired["zero_volume_observations"].sum()),
            int(selected_quality["zero_volume_observations"].sum()),
            original_raw_rows,
            filtered_raw_rows,
            int(acquired["zero_volume_observations"].gt(0).sum()),
            int(selected_quality["zero_volume_observations"].gt(0).sum()),
        ),
        (
            "Extreme diagnostic returns",
            int(acquired["extreme_return_rows"].sum()),
            int(selected_quality["extreme_return_rows"].sum()),
            len(pd.read_parquet(project_path(config["inputs"]["diagnostic_returns"]), columns=["date"])),
            len(selected_diagnostics),
            int(acquired["extreme_return_rows"].gt(0).sum()),
            int(selected_quality["extreme_return_rows"].gt(0).sum()),
        ),
        (
            "Unresolved return discontinuities",
            len(all_unresolved),
            len(selected_unresolved),
            int(acquired["extreme_return_rows"].sum()),
            int(selected_quality["extreme_return_rows"].sum()),
            all_unresolved["security_id"].nunique(),
            selected_unresolved["security_id"].nunique(),
        ),
        (
            "Corporate-action discrepancy events",
            len(reconciliation),
            len(selected_reconciliation),
            original_events,
            filtered_events,
            reconciliation["security_id"].nunique(),
            selected_reconciliation["security_id"].nunique(),
        ),
        (
            "Quarantine-candidate rows/windows",
            len(quarantine),
            len(selected_quarantine),
            len(base_all),
            len(selected_base),
            quarantine["security_id"].nunique(),
            selected_quarantine["security_id"].nunique(),
        ),
    ]
    rows = []
    for metric, original, filtered, original_denom, filtered_denom, original_sec, filtered_sec in metric_specs:
        rows.append(
            {
                "Metric": metric,
                "Original": int(original),
                "Filtered": int(filtered),
                "Original rate": (
                    100 * original / original_denom
                    if original_denom
                    else np.nan
                ),
                "Filtered rate": (
                    100 * filtered / filtered_denom
                    if filtered_denom
                    else np.nan
                ),
                "Original affected securities": int(original_sec),
                "Filtered affected securities": int(filtered_sec),
                "Filtered share of original": (
                    f"{100 * filtered / original:.2f}%" if original else "Unknown"
                ),
            }
        )
    return pd.DataFrame(rows)


def _figures(
    year: pd.DataFrame,
    compare: pd.DataFrame,
    figure_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(
        year["year"],
        year["validated_rows"],
        marker="o",
        color=COLORS["navy"],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Validated rows")
    ax.set_title("Post-filter validated observations by year")
    ax.grid(alpha=0.6)
    fig.savefig(figure_dir / "post_filter_observations_by_year.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    selected = compare[
        compare["Metric"].isin(
            [
                "Missing adjusted-close rows (raw)",
                "Missing-volume rows (raw)",
                "Zero-volume rows",
                "Extreme diagnostic returns",
                "Quarantine-candidate rows/windows",
            ]
        )
    ]
    y = np.arange(len(selected))
    ax.barh(y - 0.18, selected["Original"], height=0.34, color=COLORS["slate"], label="original")
    ax.barh(y + 0.18, selected["Filtered"], height=0.34, color=COLORS["gold"], label="filtered")
    ax.set_yticks(y, selected["Metric"])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Rows or candidates (log scale)")
    ax.legend()
    ax.set_title("Before-and-after data-quality counts")
    fig.savefig(figure_dir / "post_filter_before_after.pdf")
    plt.close(fig)


def run(config_path: str | Path) -> dict[str, Any]:
    configure_logging(PROJECT_ROOT / "logs" / "post_filter_integrity.log")
    config = load_config(project_path(config_path))
    filtered_root = project_path(config["outputs"]["root"])
    quality_dir = filtered_root / "quality"
    tables_dir = project_path(config["outputs"]["tables"])
    figures_dir = project_path(config["outputs"]["figures"])
    quality_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    selected_master_path = filtered_root / "selected_security_master.parquet"
    selected_daily_path = filtered_root / "selected_equity_daily.parquet"
    if not selected_master_path.exists() or not selected_daily_path.exists():
        raise FileNotFoundError(
            "Liquidity-screen outputs are not ready: "
            f"{selected_master_path}, {selected_daily_path}"
        )

    protected_paths = [
        project_path(config["inputs"]["base"]),
        project_path(config["inputs"]["standardized_raw"]),
        project_path(config["inputs"]["security_master"]),
    ]
    before_hashes = {str(path): _sha256(path) for path in protected_paths}

    selected_master = pd.read_parquet(selected_master_path)
    selected_ids = set(selected_master["security_id"].astype(str))
    selected_daily = pd.read_parquet(selected_daily_path)
    selected_daily["date"] = pd.to_datetime(selected_daily["date"])

    raw = pd.read_parquet(
        project_path(config["inputs"]["standardized_raw"]),
        columns=[
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
        ],
    )
    raw["date"] = pd.to_datetime(raw["date"])
    selected_raw = raw[raw["security_id"].astype(str).isin(selected_ids)].copy()

    diagnostics = pd.read_parquet(project_path(config["inputs"]["diagnostic_returns"]))
    diagnostics["date"] = pd.to_datetime(diagnostics["date"])
    selected_diagnostics = diagnostics[
        diagnostics["security_id"].astype(str).isin(selected_ids)
    ].copy()
    quality_all = pd.read_csv(
        project_path(config["inputs"]["quality_by_security"]), low_memory=False
    )
    selected_quality = quality_all[
        quality_all["security_id"].astype(str).isin(selected_ids)
    ].copy()

    identity_path = PROJECT_ROOT / "data/investigation/identity/identity_validation.csv"
    identity = pd.read_csv(identity_path, low_memory=False)
    selected_identity = identity[
        identity["security_id"].astype(str).isin(selected_ids)
    ].copy()
    selected_identity_review_candidates = identity_review_candidates(
        identity, selected_ids
    )
    _write_csv(
        selected_identity_review_candidates,
        quality_dir / "identity_review_candidates.csv",
    )

    reconciliation = pd.read_csv(
        PROJECT_ROOT
        / "data/investigation/anomalies/corporate_action_split_reconciliation.csv",
        low_memory=False,
    )
    reconciliation = pd.concat(
        [
            reconciliation,
            pd.read_csv(
                PROJECT_ROOT
                / "data/investigation/anomalies/corporate_action_dividend_reconciliation.csv",
                low_memory=False,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    quarantine = pd.read_csv(
        PROJECT_ROOT
        / "data/investigation/anomalies/analysis_quarantine_candidates.csv",
        low_memory=False,
    )
    all_extreme_cases = pd.read_csv(
        PROJECT_ROOT / "data/investigation/anomalies/extreme_return_cases.csv",
        low_memory=False,
    )
    selected_unresolved_cases = all_extreme_cases[
        all_extreme_cases["security_id"].astype(str).isin(selected_ids)
        & all_extreme_cases["classification"].eq(
            "unresolved_extreme_discontinuity"
        )
    ].copy()

    missing_counts = _raw_metrics(raw, pd.read_parquet(
        project_path(config["inputs"]["base"]),
        columns=["date", "security_id"],
    ), selected_ids)

    coverage_rows = []
    interval_rows = []
    gap_rows = []
    calendar = xcals.get_calendar(config["project"]["calendar"])
    sessions = calendar.sessions_in_range(
        "2016-01-01", "2025-12-31"
    )
    coverage = _selection_coverage(selected_master, selected_daily, sessions)
    for row in selected_master.itertuples(index=False):
        group = selected_raw[selected_raw["security_id"].eq(row.security_id)]
        interval_rows.extend(
            missing_intervals(
                group,
                "adj_close",
                security_id=row.security_id,
                ticker=row.ticker_2023,
            )
        )
        interval_rows.extend(
            missing_intervals(
                group,
                "volume",
                security_id=row.security_id,
                ticker=row.ticker_2023,
            )
        )
        observed = pd.DatetimeIndex(
            selected_daily.loc[
                selected_daily["security_id"].eq(row.security_id), "date"
            ].unique()
        )
        gap_rows.extend(
            calendar_gap_intervals(
                observed,
                sessions,
                security_id=row.security_id,
                ticker=row.ticker_2023,
            )
        )
    intervals = pd.DataFrame(interval_rows)
    gaps = pd.DataFrame(gap_rows)

    selected_quality = selected_quality.merge(
        coverage[
            [
                "security_id",
                "leading_sessions_unknown",
                "trailing_sessions_unknown",
                "incomplete_2016_2025_interval",
            ]
        ],
        on="security_id",
        how="left",
    )
    _write_csv(selected_quality, quality_dir / "quality_by_security.csv")
    _write_csv(coverage, quality_dir / "security_coverage.csv")
    _write_csv(intervals, quality_dir / "missing_value_intervals.csv")
    _write_csv(gaps, quality_dir / "calendar_gap_intervals.csv")

    selected_daily_year = selected_daily.assign(
        year=selected_daily["date"].dt.year
    )
    selected_raw_year = selected_raw.assign(year=selected_raw["date"].dt.year)
    valid_by_year = (
        selected_daily_year.groupby("year")
        .agg(
            securities_with_valid_observations=("security_id", "nunique"),
            validated_rows=("security_id", "size"),
        )
        .reset_index()
    )
    raw_by_year = (
        selected_raw_year.groupby("year")
        .agg(
            raw_rows=("security_id", "size"),
            missing_adj_close_rows=("adj_close", lambda s: int(s.isna().sum())),
            missing_volume_rows=("volume", lambda s: int(s.isna().sum())),
            zero_volume_rows=("volume", lambda s: int(s.eq(0).sum())),
        )
        .reset_index()
    )
    quality_year = raw_by_year.merge(valid_by_year, on="year", how="outer").fillna(0)
    _write_csv(quality_year, quality_dir / "quality_by_year.csv")

    multi = selected_master.groupby("cik", dropna=False).filter(lambda g: len(g) > 1)
    multi_report = (
        multi.groupby("cik", dropna=False)
        .agg(
            securities=("security_id", "nunique"),
            tickers=("ticker_2023", lambda s: ", ".join(sorted(s.astype(str)))),
            company_names=("company_name", lambda s: " | ".join(sorted(set(s.astype(str))))),
            exchanges=("exchange", lambda s: ", ".join(sorted(set(s.astype(str))))),
            security_types=("security_type", lambda s: ", ".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
        .sort_values(["securities", "cik"], ascending=[False, True])
    )
    _write_csv(multi_report, quality_dir / "multi_security_issuers.csv")

    identity_metrics = pd.DataFrame(
        [
            {
                "check": "selected_security_ids_unique",
                "value": int(selected_master["security_id"].is_unique),
            },
            {
                "check": "selected_security_date_keys_unique",
                "value": int(
                    not selected_daily.duplicated(["security_id", "date"]).any()
                ),
            },
            {
                "check": "selected_ids_match_identity_valid",
                "value": int(
                    set(selected_identity.loc[selected_identity["identity_valid"], "security_id"].astype(str))
                    == selected_ids
                ),
            },
            {
                "check": "selected_identity_review_candidates",
                "value": int(
                    selected_identity_review_candidates["security_id"].nunique()
                ),
            },
            {
                "check": "selected_securities",
                "value": len(selected_master),
            },
            {
                "check": "selected_distinct_issuers",
                "value": selected_master["cik"].nunique(),
            },
            {
                "check": "selected_multi_security_issuers",
                "value": len(multi_report),
            },
        ]
    )
    _write_csv(identity_metrics, quality_dir / "identity_checks.csv")

    compare = _before_after(
        config,
        selected_ids,
        selected_master,
        selected_daily,
        selected_raw,
        selected_diagnostics,
        selected_quality,
        quality_all,
        reconciliation,
        quarantine,
    )
    _write_csv(compare, quality_dir / "before_after_metrics.csv")
    _write_csv(compare, tables_dir / "before_after_metrics.csv")
    _write_tex(
        compare.rename(
            columns={
                "Metric": "Metric",
                "Original": "Original",
                "Filtered": "Filtered",
                "Filtered share of original": "Share",
            }
        ),
        tables_dir / "table_before_after.tex",
        "lrrr",
    )
    _write_csv(quality_year, tables_dir / "post_filter_quality_by_year.csv")
    _write_tex(
        quality_year,
        tables_dir / "table_post_filter_quality_by_year.tex",
        "r" * len(quality_year.columns),
    )
    _write_csv(multi_report, tables_dir / "multi_security_issuers.csv")

    filtered_issues = pd.read_csv(
        project_path(config["inputs"]["validation_issues"]), low_memory=False
    )
    filtered_issues = filtered_issues[
        filtered_issues["security_id"].astype(str).isin(selected_ids)
    ].copy()
    filtered_issues.insert(0, "data_scope", "filtered_cohort")
    _write_csv(filtered_issues, quality_dir / "filtered_data_issues.csv")

    _figures(quality_year, compare, figures_dir)

    after_hashes = {str(path): _sha256(path) for path in protected_paths}
    hashes_unchanged = before_hashes == after_hashes
    selected_config = json.loads(json.dumps(config))
    summary = {
        "selected_securities": int(len(selected_master)),
        "selected_distinct_issuers": int(selected_master["cik"].nunique()),
        "validated_rows": int(len(selected_daily)),
        "raw_rows": int(missing_counts["raw_rows"]),
        "raw_rows_not_in_base": int(missing_counts["raw_rows_not_in_base"]),
        "placeholder_all_nan_rows": int(missing_counts["placeholder_all_nan_rows"]),
        "missing_adj_close_rows": int(missing_counts["missing_adj_close_rows"]),
        "missing_volume_rows": int(missing_counts["missing_volume_rows"]),
        "zero_volume_rows": int(missing_counts["zero_volume_rows"]),
        "extreme_return_rows": int(
            selected_quality["extreme_return_rows"].sum()
        ),
        "unresolved_extreme_cases": int(len(selected_unresolved_cases)),
        "unresolved_extreme_securities": int(
            selected_unresolved_cases["security_id"].nunique()
        ),
        "internal_calendar_gaps": int(len(gaps)),
        "missing_value_intervals": int(len(intervals)),
        "incomplete_history_securities": int(
            coverage["incomplete_2016_2025_interval"].sum()
        ),
        "multi_security_issuers": int(len(multi_report)),
        "identity_review_candidates": int(
            identity_metrics.loc[
                identity_metrics["check"].eq(
                    "selected_identity_review_candidates"
                ),
                "value",
            ].iloc[0]
        ),
        "identity_review_candidate_tickers": sorted(
            selected_identity_review_candidates["ticker_2023"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "protected_inputs_unchanged": bool(hashes_unchanged),
        "input_sha256_before": before_hashes,
        "input_sha256_after": after_hashes,
        "config": selected_config,
    }
    atomic_write_json(summary, quality_dir / "post_filter_integrity_summary.json")
    (PROJECT_ROOT / "docs/post_filter_integrity.md").write_text(
        f"""# Post-filter integrity assessment

The filtered cohort contains {len(selected_master):,} securities representing
{selected_master['cik'].nunique():,} distinct SEC issuers. The retained base
contains {len(selected_daily):,} validated stock-day rows.

Raw selected-cohort reconciliation finds {missing_counts['raw_rows']:,} raw
rows, {missing_counts['raw_rows_not_in_base']:,} raw rows excluded from the
validated base, and {missing_counts['placeholder_all_nan_rows']:,} all-NaN
price placeholders. Missing adjusted-close and volume rows are retained in the
audit archive and are never filled with zero.

The selected cohort has {int(coverage['incomplete_2016_2025_interval'].sum()):,}
securities whose observed history does not span the full calendar. Because the
master lacks verified listing and delisting dates, leading and trailing periods
remain unknown rather than classified as structural.

The identity review flag overlaps {len(selected_identity_review_candidates):,}
selected securities: {', '.join(sorted(selected_identity_review_candidates['ticker_2023'].astype(str)))}.
Their current action is `base_rows_only`, not confirmed cross-ticker
contamination; the evidence file is retained in `identity_review_candidates.csv`.

Protected source hashes were unchanged after the audit: {hashes_unchanged}.
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs/liquidity_filter.yaml")
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
