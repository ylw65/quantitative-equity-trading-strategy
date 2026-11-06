"""Build the fixed 2023 ADTV20 liquidity cohort without mutating source data."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"),
)

import exchange_calendars as xcals
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


LOGGER = logging.getLogger("us_equity_data.liquidity")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    target = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def final_sessions(
    calendar_name: str,
    window_end: str,
    count: int,
) -> pd.DatetimeIndex:
    """Return the final `count` sessions ending on `window_end`."""
    calendar = xcals.get_calendar(calendar_name)
    end = pd.Timestamp(window_end).normalize()
    if not calendar.is_session(end):
        raise ValueError(f"{window_end} is not a {calendar_name} trading session")
    start = end - pd.DateOffset(years=1)
    sessions = calendar.sessions_in_range(start, end)
    if len(sessions) < count:
        raise ValueError("Calendar window contains fewer sessions than requested")
    return sessions[-count:]


def daily_dollar_volume(close: Any, volume: Any) -> np.ndarray:
    """Calculate valid row-level dollar volume without imputing missing data."""
    close_values = np.asarray(close, dtype="float64")
    volume_values = np.asarray(volume, dtype="float64")
    valid = (
        np.isfinite(close_values)
        & np.isfinite(volume_values)
        & (close_values > 0)
        & (volume_values >= 0)
    )
    result = np.full(close_values.shape, np.nan, dtype="float64")
    result[valid] = close_values[valid] * volume_values[valid]
    return result


def adtv20(daily_values: pd.Series, expected_sessions: int = 20) -> float:
    """Return ADTV20 only when every specified session has a valid value."""
    values = pd.to_numeric(daily_values, errors="coerce")
    valid = values.notna() & np.isfinite(values) & values.ge(0)
    if int(valid.sum()) != expected_sessions or len(values) != expected_sessions:
        return float("nan")
    return float(values.mean())


def classify_window(
    valid_sessions: int,
    expected_sessions: int,
    adtv20_usd: float,
    threshold_usd: float,
    *,
    unresolved_price_volume: bool = False,
) -> str:
    """Apply the fixed screen using mutually exclusive statuses."""
    if unresolved_price_volume:
        return "unresolved_price_volume_issue"
    if valid_sessions != expected_sessions:
        return "insufficient_liquidity_window_data"
    if np.isfinite(adtv20_usd) and adtv20_usd >= threshold_usd:
        return "selected"
    if np.isfinite(adtv20_usd):
        return "excluded_below_threshold"
    return "unresolved_price_volume_issue"


def future_split_product(
    splits: pd.DataFrame,
    observation_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Return product of split factors strictly after an observation date."""
    if splits.empty:
        return 1.0
    factors = pd.to_numeric(
        splits.loc[
            splits["date"].gt(observation_date)
            & splits["date"].le(end_date),
            "stock_splits",
        ],
        errors="coerce",
    )
    if factors.isna().any() or factors.le(0).any():
        return float("nan")
    return float(factors.prod()) if len(factors) else 1.0


def _build_window_grid(
    acquired_master: pd.DataFrame,
    raw_window: pd.DataFrame,
    splits: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    split_end: pd.Timestamp,
) -> pd.DataFrame:
    """Create an explicit row for every acquired security-session combination."""
    grid = pd.MultiIndex.from_product(
        [acquired_master["security_id"], sessions],
        names=["security_id", "date"],
    ).to_frame(index=False)
    grid = grid.merge(
        acquired_master[
            [
                "security_id",
                "ticker_2023",
                "yahoo_ticker",
                "company_name",
                "exchange",
                "security_type",
                "cik",
            ]
        ],
        on="security_id",
        how="left",
        validate="many_to_one",
    )
    vendor = raw_window[
        [
            "security_id",
            "date",
            "close",
            "volume",
            "stock_splits",
            "source_file",
        ]
    ].copy()
    if vendor.duplicated(["security_id", "date"]).any():
        raise ValueError("Duplicate security-date rows in liquidity source window")
    grid = grid.merge(
        vendor,
        on=["security_id", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    grid["vendor_row_present"] = grid["_merge"].eq("both")
    grid = grid.drop(columns="_merge")

    split_groups = {
        security_id: group.sort_values("date")
        for security_id, group in splits.groupby("security_id", sort=False)
    }
    grid["future_split_factor"] = [
        future_split_product(
            split_groups.get(security_id, pd.DataFrame()),
            date,
            split_end,
        )
        for security_id, date in zip(grid["security_id"], grid["date"])
    ]
    grid["future_split_events"] = [
        (
            0
            if security_id not in split_groups
            else int(
                split_groups[security_id]["date"].gt(date).mul(
                    split_groups[security_id]["date"].le(split_end)
                ).sum()
            )
        )
        for security_id, date in zip(grid["security_id"], grid["date"])
    ]
    grid["vendor_close_usd"] = pd.to_numeric(grid["close"], errors="coerce")
    grid["reported_volume"] = pd.to_numeric(grid["volume"], errors="coerce")
    grid["valid_close"] = (
        grid["vendor_row_present"]
        & np.isfinite(grid["vendor_close_usd"])
        & grid["vendor_close_usd"].gt(0)
    )
    grid["valid_volume"] = (
        grid["vendor_row_present"]
        & np.isfinite(grid["reported_volume"])
        & grid["reported_volume"].ge(0)
    )
    grid["price_volume_compatible"] = (
        np.isfinite(grid["future_split_factor"])
        & grid["future_split_factor"].gt(0)
    )
    grid["compatible_close_usd"] = np.where(
        grid["valid_close"] & grid["price_volume_compatible"],
        grid["vendor_close_usd"] * grid["future_split_factor"],
        np.nan,
    )
    grid["daily_dollar_volume_usd"] = daily_dollar_volume(
        grid["compatible_close_usd"],
        grid["reported_volume"].where(grid["valid_volume"], np.nan),
    )
    grid["observation_valid"] = grid["daily_dollar_volume_usd"].notna()
    grid["session_index"] = grid["date"].map(
        {session.normalize(): position + 1 for position, session in enumerate(sessions)}
    )
    return grid.sort_values(["security_id", "date"], kind="stable").reset_index(
        drop=True
    )


def _screen_grid(
    grid: pd.DataFrame,
    master: pd.DataFrame,
    threshold: float,
    expected_sessions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for security_id, group in grid.groupby("security_id", sort=True):
        metadata = group.iloc[0]
        valid_count = int(group["observation_valid"].sum())
        adtv = adtv20(
            group["daily_dollar_volume_usd"],
            expected_sessions=expected_sessions,
        )
        unresolved = bool(
            len(group) != expected_sessions
            or (~np.isfinite(group["future_split_factor"])).any()
            or group["future_split_factor"].le(0).any()
        )
        status = classify_window(
            valid_count,
            expected_sessions,
            adtv,
            threshold,
            unresolved_price_volume=unresolved,
        )
        rows.append(
            {
                "security_id": security_id,
                "ticker_2023": metadata["ticker_2023"],
                "yahoo_ticker": metadata["yahoo_ticker"],
                "company_name": metadata["company_name"],
                "exchange": metadata["exchange"],
                "security_type": metadata["security_type"],
                "cik": metadata["cik"],
                "acquisition_status": "acquired",
                "screen_status": status,
                "window_sessions_expected": expected_sessions,
                "window_rows_observed": int(group["vendor_row_present"].sum()),
                "valid_window_sessions": valid_count,
                "missing_window_sessions": expected_sessions - valid_count,
                "invalid_close_rows": int(
                    (group["vendor_row_present"] & ~group["valid_close"]).sum()
                ),
                "invalid_volume_rows": int(
                    (group["vendor_row_present"] & ~group["valid_volume"]).sum()
                ),
                "zero_volume_window_rows": int(
                    (group["valid_volume"] & group["reported_volume"].eq(0)).sum()
                ),
                "future_split_events_restored": int(
                    group["future_split_events"].max()
                ),
                "future_split_factor_min": float(
                    group["future_split_factor"].min()
                ),
                "future_split_factor_max": float(
                    group["future_split_factor"].max()
                ),
                "adtv20_usd": adtv,
                "threshold_usd": threshold,
                "eligible": status == "selected",
                "price_volume_rule": (
                    "vendor_nominal_close_times_future_split_product_times_"
                    "reported_volume"
                ),
                "price_volume_issue": unresolved,
            }
        )
    result = pd.DataFrame(rows)
    result["selected"] = result["screen_status"].eq("selected")
    return result.merge(
        master[["security_id", "permanent_id", "reference_date_verified"]],
        on="security_id",
        how="left",
        validate="one_to_one",
    )


def _all_universe_screen(screen: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "security_id",
        "ticker_2023",
        "yahoo_ticker",
        "company_name",
        "exchange",
        "security_type",
        "cik",
        "acquisition_status",
        "screen_status",
        "window_sessions_expected",
        "window_rows_observed",
        "valid_window_sessions",
        "missing_window_sessions",
        "invalid_close_rows",
        "invalid_volume_rows",
        "zero_volume_window_rows",
        "future_split_events_restored",
        "future_split_factor_min",
        "future_split_factor_max",
        "adtv20_usd",
        "threshold_usd",
        "selected",
        "price_volume_issue",
    ]
    unavailable = master.loc[
        ~master["security_id"].isin(set(screen["security_id"]))
    ].copy()
    unavailable["acquisition_status"] = "not_acquired"
    unavailable["screen_status"] = "not_acquired"
    unavailable["window_sessions_expected"] = 20
    unavailable["window_rows_observed"] = 0
    unavailable["valid_window_sessions"] = 0
    unavailable["missing_window_sessions"] = 20
    unavailable["invalid_close_rows"] = 0
    unavailable["invalid_volume_rows"] = 0
    unavailable["zero_volume_window_rows"] = 0
    unavailable["future_split_events_restored"] = 0
    unavailable["future_split_factor_min"] = np.nan
    unavailable["future_split_factor_max"] = np.nan
    unavailable["adtv20_usd"] = np.nan
    unavailable["threshold_usd"] = np.nan
    unavailable["selected"] = False
    unavailable["price_volume_issue"] = False
    all_screen = pd.concat(
        [screen.reindex(columns=columns), unavailable.reindex(columns=columns)],
        ignore_index=True,
    )
    return all_screen.sort_values(
        ["selected", "screen_status", "ticker_2023"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _composition(
    acquired_master: pd.DataFrame,
    selected_master: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in ("exchange", "security_type"):
        before = acquired_master[dimension].value_counts()
        after = selected_master[dimension].value_counts()
        values = sorted(set(before.index).union(after.index))
        for value in values:
            before_count = int(before.get(value, 0))
            after_count = int(after.get(value, 0))
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "acquired_securities": before_count,
                    "selected_securities": after_count,
                    "retention_rate": (
                        after_count / before_count if before_count else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _quantiles(screen: pd.DataFrame) -> pd.DataFrame:
    complete = screen.loc[
        screen["screen_status"].isin({"selected", "excluded_below_threshold"}),
        "adtv20_usd",
    ].dropna()
    stats = complete.describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    return stats.rename("adtv20_usd").rename_axis("statistic").reset_index()


def _sensitivity(screen: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    complete = screen["adtv20_usd"].dropna()
    return pd.DataFrame(
        [
            {
                "threshold_usd": threshold,
                "securities_at_or_above": int(complete.ge(threshold).sum()),
                "binding": threshold == 100_000_000,
            }
            for threshold in thresholds
        ]
    )


def _figure(screen: pd.DataFrame, threshold: float, path: Path) -> None:
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
    complete = screen["adtv20_usd"].dropna()
    positive = complete.loc[complete.gt(0)].sort_values()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    if not positive.empty:
        ecdf = np.arange(1, len(positive) + 1) / len(positive)
        ax.plot(positive, ecdf, color="#19324D", linewidth=1.4)
    ax.axvline(
        threshold,
        color="#A45D4F",
        linestyle="--",
        label=f"Threshold: USD {threshold:,.0f}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("ADTV20 (USD, log scale)")
    ax.set_ylabel("Empirical cumulative share")
    ax.set_title("Liquidity-screen ECDF")
    ax.grid(alpha=0.6)
    ax.legend()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def run(config_path: str | Path) -> dict[str, Any]:
    """Run the deterministic fixed-cohort liquidity filter."""
    config_path = project_path(config_path)
    config = load_config(config_path)
    configure_logging(project_path("logs/liquidity_filter.log"))
    project = config["project"]
    input_paths = {name: project_path(path) for name, path in config["inputs"].items()}
    output_root = project_path(config["outputs"]["root"])
    table_dir = project_path(config["outputs"]["tables"])
    figure_dir = project_path(config["outputs"]["figures"])
    output_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        name: _sha256(path) for name, path in input_paths.items() if path.exists()
    }
    raw_file_count = len(list(project_path("data/raw/yahoo").glob("*.parquet")))

    master = pd.read_parquet(input_paths["security_master"])
    base = pd.read_parquet(input_paths["base"])
    raw = pd.read_parquet(
        input_paths["standardized_raw"],
        columns=[
            "date",
            "security_id",
            "ticker",
            "close",
            "volume",
            "stock_splits",
            "source_file",
        ],
    )
    manifest = pd.read_csv(input_paths["download_manifest"], low_memory=False)
    acquired_ids = set(
        manifest.loc[
            manifest["download_status"].eq("success"), "security_id"
        ].astype(str)
    )
    acquired_master = master.loc[
        master["security_id"].astype(str).isin(acquired_ids)
    ].copy()
    sessions = final_sessions(
        project["calendar"],
        project["window_end"],
        int(project["window_sessions"]),
    )
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw_window = raw.loc[
        raw["security_id"].astype(str).isin(acquired_ids)
        & raw["date"].isin(sessions)
    ].copy()
    splits = raw.loc[
        raw["security_id"].astype(str).isin(acquired_ids)
        & raw["stock_splits"].gt(0)
        & raw["date"].le(pd.Timestamp(project["future_split_adjustment"]["end_date"]))
    ][["security_id", "date", "stock_splits"]].copy()

    grid = _build_window_grid(
        acquired_master,
        raw_window,
        splits,
        sessions,
        pd.Timestamp(project["future_split_adjustment"]["end_date"]),
    )
    screen = _screen_grid(
        grid,
        master,
        float(project["threshold_usd"]),
        int(project["window_sessions"]),
    )
    all_screen = _all_universe_screen(screen, master)
    selected = screen.loc[screen["selected"]].copy()
    excluded = all_screen.loc[~all_screen["selected"]].copy()
    selected_ids = set(selected["security_id"].astype(str))
    selected_master = (
        master.loc[master["security_id"].astype(str).isin(selected_ids)]
        .merge(
            screen[
                [
                    "security_id",
                    "screen_status",
                    "window_sessions_expected",
                    "valid_window_sessions",
                    "zero_volume_window_rows",
                    "future_split_events_restored",
                    "adtv20_usd",
                    "threshold_usd",
                    "price_volume_rule",
                    "price_volume_issue",
                ]
            ],
            on="security_id",
            how="left",
            validate="one_to_one",
        )
        .copy()
    )
    selected_daily = base.loc[
        base["security_id"].astype(str).isin(selected_ids)
    ].copy()
    selected_daily["pct_return"] = np.nan
    selected_daily = selected_daily.sort_values(
        ["security_id", "date"], kind="stable"
    ).reset_index(drop=True)
    if selected_daily.duplicated(["security_id", "date"]).any():
        raise ValueError("Filtered daily output contains duplicate security-date keys")

    output_root.joinpath("selected_security_master.parquet").parent.mkdir(
        parents=True, exist_ok=True
    )
    _atomic_parquet(
        selected_master.sort_values("security_id").reset_index(drop=True),
        output_root / "selected_security_master.parquet",
    )
    _atomic_parquet(
        selected_daily,
        output_root / "selected_equity_daily.parquet",
    )
    _atomic_csv(all_screen, output_root / "liquidity_screen_all.csv")
    _atomic_csv(
        selected.sort_values("adtv20_usd", ascending=False),
        output_root / "liquidity_screen_selected.csv",
    )
    _atomic_csv(
        excluded.sort_values(["screen_status", "ticker_2023"]),
        output_root / "liquidity_screen_excluded.csv",
    )
    _atomic_parquet(
        grid,
        output_root / "liquidity_window_observations.parquet",
    )

    composition = _composition(acquired_master, selected_master)
    quantiles = _quantiles(screen)
    sensitivity = _sensitivity(
        screen,
        [float(value) for value in project["sensitivity_thresholds_usd"]],
    )
    status_counts = screen["screen_status"].value_counts().to_dict()
    flow = pd.DataFrame(
        [
            {"stage": "SEC-derived historical universe", "securities": len(master)},
            {"stage": "Yahoo-acquired securities", "securities": len(acquired_master)},
            {
                "stage": "Complete valid 20-session window",
                "securities": int(
                    screen["screen_status"].isin(
                        {"selected", "excluded_below_threshold"}
                    ).sum()
                ),
            },
            {"stage": "ADTV20 >= USD 100 million", "securities": len(selected)},
            {"stage": "Filtered 2016-2025 dataset", "securities": len(selected)},
        ]
    )
    before_after = pd.DataFrame(
        [
            {"metric": "securities", "before": len(acquired_master), "after": len(selected)},
            {"metric": "stock_day_rows", "before": len(base), "after": len(selected_daily)},
            {"metric": "distinct_issuers", "before": acquired_master["cik"].nunique(), "after": selected_master["cik"].nunique()},
        ]
    )
    _atomic_csv(composition, table_dir / "liquidity_composition.csv")
    _atomic_csv(quantiles, table_dir / "liquidity_adtv_quantiles.csv")
    _atomic_csv(sensitivity, table_dir / "liquidity_sensitivity.csv")
    _atomic_csv(flow, table_dir / "liquidity_selection_flow.csv")
    _atomic_csv(before_after, table_dir / "liquidity_before_after_counts.csv")
    _figure(
        screen,
        float(project["threshold_usd"]),
        figure_dir / "liquidity_adtv20_ecdf.pdf",
    )

    selected_adtv = selected["adtv20_usd"]
    metadata = {
        "generated_at_utc": utc_now(),
        "experiment": project["name"],
        "as_of_date": project["as_of_date"],
        "calendar": project["calendar"],
        "window_end": project["window_end"],
        "window_sessions": [session.date().isoformat() for session in sessions],
        "window_session_count": len(sessions),
        "threshold_usd": float(project["threshold_usd"]),
        "threshold_inclusive": True,
        "price_volume_method": project["future_split_adjustment"],
        "selection_flow": flow.to_dict("records"),
        "screen_status_counts": status_counts,
        "universe_securities": len(master),
        "acquired_securities": len(acquired_master),
        "complete_window_securities": int(
            screen["screen_status"].isin(
                {"selected", "excluded_below_threshold"}
            ).sum()
        ),
        "selected_securities": len(selected),
        "selected_observations": len(selected_daily),
        "selected_distinct_issuers": int(selected_master["cik"].nunique()),
        "acquired_retention_rate": len(selected) / len(acquired_master)
        if len(acquired_master)
        else None,
        "observation_retention_rate": len(selected_daily) / len(base)
        if len(base)
        else None,
        "selected_adtv20_usd": {
            "min": float(selected_adtv.min()) if len(selected_adtv) else None,
            "median": float(selected_adtv.median()) if len(selected_adtv) else None,
            "max": float(selected_adtv.max()) if len(selected_adtv) else None,
        },
        "sensitivity_thresholds": sensitivity.to_dict("records"),
        "source_hashes_sha256": input_hashes,
        "raw_yahoo_file_count_at_run": raw_file_count,
        "research_constraints": config["research_constraints"],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "exchange_calendars": xcals.__version__,
        },
    }
    atomic_write_json(metadata, output_root / "filter_metadata.json")
    (PROJECT_ROOT / "docs" / "liquidity_filter_methodology.md").write_text(
        f"""# Liquidity-filter methodology

The fixed screen uses the final {len(sessions)} XNYS sessions ending
{project['window_end']}. A row is valid only for a finite positive nominal
close and finite nonnegative reported share volume. Missing values and absent
sessions are not substituted or filled. A valid zero-volume row contributes
zero dollar volume.

Yahoo's nominal Close is split-back-adjusted while Volume is treated as
reported contemporaneous share volume. Each window Close is restored to the
observation-date share basis by multiplying future split factors dated after
that observation through {project['future_split_adjustment']['end_date']}.
Daily dollar volume is then compatible Close times reported Volume. This uses
future corporate-action metadata only to interpret historical vendor
adjustments, never future prices, volumes, or returns.

The inclusive threshold is USD {project['threshold_usd']:,}. Of
{len(acquired_master):,} acquired securities, {len(selected):,} selected into the
fixed 2016-2025 panel containing {len(selected_daily):,} validated rows. All raw
and previously validated files remain unchanged.
""",
        encoding="utf-8",
    )
    return {
        "universe": len(master),
        "acquired": len(acquired_master),
        "complete_window": metadata["complete_window_securities"],
        "selected": len(selected),
        "selected_observations": len(selected_daily),
        "screen_status_counts": status_counts,
        "output_root": str(output_root),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/liquidity_filter.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
