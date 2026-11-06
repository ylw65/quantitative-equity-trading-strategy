"""Investigate zero-volume concentration and split-related volume behavior."""

from __future__ import annotations

import argparse
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


def _write_tex(frame: pd.DataFrame, path: Path, align: str = "lrr") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            if isinstance(value, float):
                if np.isnan(value):
                    cells.append("Unknown")
                elif abs(value) < 100 and value != int(value):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(f"{int(round(value)):,}")
            elif isinstance(value, (int, np.integer)):
                integer = int(value)
                cells.append(
                    str(integer)
                    if 1900 <= integer <= 2100
                    else f"{integer:,}"
                )
            else:
                cells.append(str(value).replace("_", r"\_").replace("&", r"\&"))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def longest_true_run(values: pd.Series) -> int:
    """Return the longest consecutive run of truthy values."""
    if values.empty:
        return 0
    values = values.fillna(False).astype(bool).to_numpy()
    if not values.any():
        return 0
    groups = np.diff(np.r_[False, values, False].astype(int))
    starts = np.flatnonzero(groups == 1)
    ends = np.flatnonzero(groups == -1)
    return int((ends - starts).max())


def zero_concentration(zero_by_security: pd.Series) -> dict[str, float | int]:
    """Summarize concentration of zero-volume rows across securities."""
    nonzero = zero_by_security.loc[zero_by_security.gt(0)].sort_values(
        ascending=False
    )
    total = int(nonzero.sum())
    securities = int(len(nonzero))
    if total:
        cumulative = nonzero.cumsum()
        half_count = int((cumulative < total / 2).sum() + 1)
    else:
        half_count = 0
    return {
        "securities_with_zero_volume": securities,
        "zero_volume_rows": total,
        "securities_for_first_half": half_count,
        "max_zero_rows_one_security": int(nonzero.max()) if securities else 0,
    }


def _per_security(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (security_id, ticker), group in raw.groupby(
        ["security_id", "ticker"], sort=True
    ):
        group = group.sort_values("date", kind="stable")
        zero = group["volume"].eq(0)
        positive = group["volume"].gt(0)
        zero_dates = group.loc[zero, "date"]
        positive_dates = group.loc[positive, "date"]
        rows.append(
            {
                "security_id": security_id,
                "ticker": ticker,
                "first_date": group["date"].min().date().isoformat(),
                "last_date": group["date"].max().date().isoformat(),
                "total_rows": len(group),
                "positive_volume_rows": int(positive.sum()),
                "zero_volume_rows": int(zero.sum()),
                "zero_volume_rate": float(zero.mean()),
                "first_zero_date": (
                    zero_dates.min().date().isoformat() if len(zero_dates) else ""
                ),
                "last_zero_date": (
                    zero_dates.max().date().isoformat() if len(zero_dates) else ""
                ),
                "longest_zero_run_rows": longest_true_run(zero),
                "last_positive_volume_date": (
                    positive_dates.max().date().isoformat()
                    if len(positive_dates)
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _by_year(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.assign(year=raw["date"].dt.year, zero=raw["volume"].eq(0))
    rows = []
    for year, group in raw.groupby("year", sort=True):
        rates = group.groupby("security_id")["zero"].mean()
        rows.append(
            {
                "year": int(year),
                "securities": int(group["security_id"].nunique()),
                "observations": len(group),
                "zero_volume_rows": int(group["zero"].sum()),
                "zero_volume_rate": float(group["zero"].mean()),
                "securities_any_zero": int(rates.gt(0).sum()),
                "securities_zero_rate_gt_50pct": int(rates.gt(0.5).sum()),
                "median_security_zero_rate": float(rates.median()),
                "p90_security_zero_rate": float(rates.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _split_volume(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_groups = raw.groupby(["security_id", "ticker"], sort=True)
    for (security_id, ticker), group in split_groups:
        group = group.sort_values("date", kind="stable").reset_index(drop=True)
        event_positions = group.index[group["stock_splits"].gt(0)].tolist()
        for position in event_positions:
            if position == 0:
                continue
            before = group.iloc[max(0, position - 5) : position]["volume"]
            after = group.iloc[position + 1 : position + 6]["volume"]
            before_median = float(before.median()) if len(before) else np.nan
            after_median = float(after.median()) if len(after) else np.nan
            event_volume = float(group.at[position, "volume"])
            previous_volume = float(group.at[position - 1, "volume"])
            split_factor = float(group.at[position, "stock_splits"])
            ratio_before = (
                event_volume / before_median
                if before_median > 0
                else np.nan
            )
            base_shift = (
                after_median / before_median
                if before_median > 0 and np.isfinite(after_median)
                else np.nan
            )
            rows.append(
                {
                    "security_id": security_id,
                    "ticker": ticker,
                    "event_date": group.at[position, "date"].date().isoformat(),
                    "split_factor": split_factor,
                    "event_volume": event_volume,
                    "previous_session_volume": previous_volume,
                    "pre5_median_volume": before_median,
                    "post5_median_volume": after_median,
                    "event_volume_ratio_to_pre5": ratio_before,
                    "post5_median_ratio_to_pre5": base_shift,
                    "event_ratio_relative_to_split_factor": (
                        ratio_before / split_factor
                        if np.isfinite(ratio_before) and split_factor
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _figures(
    year: pd.DataFrame,
    security: pd.DataFrame,
    splits: pd.DataFrame,
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

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.plot(
        year["year"],
        year["zero_volume_rate"] * 100,
        marker="o",
        color=COLORS["navy"],
        label="zero-volume share",
    )
    ax.set_ylabel("Zero-volume rows (%)")
    ax.set_xlabel("Year")
    ax.set_title("Zero-volume incidence by year")
    ax.grid(alpha=0.6)
    fig.savefig(figure_dir / "volume_zero_by_year.pdf")
    plt.close(fig)

    positive = security.loc[security["zero_volume_rows"].gt(0)].sort_values(
        "zero_volume_rows", ascending=False
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    if not positive.empty:
        cumulative = positive["zero_volume_rows"].cumsum()
        share = cumulative / cumulative.iloc[-1]
        ax.plot(
            np.arange(1, len(positive) + 1),
            share,
            color=COLORS["rust"],
        )
        ax.axhline(0.5, color=COLORS["slate"], linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Securities ranked by zero-volume rows (log scale)")
        ax.set_ylabel("Cumulative share")
    ax.set_title("Concentration of zero-volume observations")
    ax.grid(alpha=0.6)
    fig.savefig(figure_dir / "volume_zero_concentration.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    rates = security["zero_volume_rate"]
    ax.hist(rates, bins=np.linspace(0, 1, 41), color=COLORS["gold"])
    ax.set_xlabel("Zero-volume share within security history")
    ax.set_ylabel("Security count")
    ax.set_title("Distribution of per-security zero-volume incidence")
    ax.grid(axis="y", alpha=0.6)
    fig.savefig(figure_dir / "volume_zero_security_distribution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    usable = splits.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(
        subset=["split_factor", "event_ratio_relative_to_split_factor"]
    )
    usable = usable[
        usable["split_factor"].gt(0)
        & usable["event_ratio_relative_to_split_factor"].gt(0)
    ]
    if not usable.empty:
        ax.scatter(
            usable["split_factor"],
            usable["event_ratio_relative_to_split_factor"],
            s=12,
            alpha=0.5,
            color=COLORS["blue"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reported split factor")
    ax.set_ylabel("Event volume ratio / split factor")
    ax.set_title("Reported volume around split events")
    ax.grid(alpha=0.6)
    fig.savefig(figure_dir / "volume_split_relationship.pdf")
    plt.close(fig)


def run(config_path: str | Path) -> dict[str, Any]:
    configure_logging(PROJECT_ROOT / "logs" / "volume_investigation.log")
    config = load_config(project_path(config_path))
    if not config:
        raise ValueError("Acquisition configuration is empty or unreadable.")

    raw_path = PROJECT_ROOT / "data" / "validated" / "standardized_raw.parquet"
    master_path = PROJECT_ROOT / "data" / "universe" / "security_master.parquet"
    raw = pd.read_parquet(
        raw_path,
        columns=[
            "date",
            "security_id",
            "ticker",
            "volume",
            "stock_splits",
            "provider",
        ],
    )
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.loc[raw["date"].notna()].sort_values(
        ["security_id", "date"], kind="stable"
    )

    output_dir = PROJECT_ROOT / "data" / "investigation" / "volume"
    table_dir = PROJECT_ROOT / "reports" / "tables" / "investigation"
    figure_dir = PROJECT_ROOT / "reports" / "figures" / "investigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    security = _per_security(raw).sort_values(
        ["zero_volume_rows", "ticker"], ascending=[False, True]
    )
    year = _by_year(raw)
    split = _split_volume(raw)
    concentration = zero_concentration(
        security.set_index("security_id")["zero_volume_rows"]
    )

    top = security.head(25).copy()
    prolonged = security.loc[
        security["zero_volume_rate"].ge(0.5)
        | security["longest_zero_run_rows"].ge(252)
    ].sort_values(
        ["longest_zero_run_rows", "zero_volume_rows"],
        ascending=False,
    )

    _write_csv(security, output_dir / "zero_volume_by_security.csv")
    _write_csv(year, output_dir / "zero_volume_by_year.csv")
    _write_csv(split, output_dir / "split_volume_events.csv")
    _write_csv(prolonged, output_dir / "prolonged_zero_volume_cases.csv")
    _write_csv(top, table_dir / "volume_zero_top_securities.csv")
    _write_csv(year, table_dir / "volume_zero_by_year.csv")
    _write_csv(
        split.sort_values(
            "event_ratio_relative_to_split_factor", ascending=False
        ).head(20),
        table_dir / "volume_split_representative.csv",
    )
    _write_tex(
        year[
            [
                "year",
                "securities",
                "zero_volume_rows",
                "zero_volume_rate",
                "securities_zero_rate_gt_50pct",
            ]
        ].copy(),
        table_dir / "table_volume_zero_by_year.tex",
        "rrrrr",
    )
    _write_tex(
        top[
            [
                "ticker",
                "zero_volume_rows",
                "zero_volume_rate",
                "longest_zero_run_rows",
                "last_positive_volume_date",
            ]
        ].copy(),
        table_dir / "table_volume_top_zero.tex",
        "lrrrl",
    )

    issues = pd.DataFrame(
        [
            {
                "issue_id": "P1-VOLUME-CONCENTRATION",
                "severity": "medium",
                "issue_type": "zero_volume_concentration",
                "status": "partially_resolved",
                "evidence": (
                    f"{concentration['zero_volume_rows']:,} zero-volume rows occur "
                    f"across {concentration['securities_with_zero_volume']:,} "
                    "securities; half of all such rows occur in the first "
                    f"{concentration['securities_for_first_half']:,} ranked securities."
                ),
                "remaining_uncertainty": (
                    "Zero volume cannot by itself distinguish illiquidity, a "
                    "suspension, an inactive vendor row, or a stale synthetic row."
                ),
                "recommended_action": (
                    "Do not use zero volume as evidence of a trade. Add explicit "
                    "future liquidity eligibility rules rather than converting zero "
                    "volume to missing."
                ),
            },
            {
                "issue_id": "P1-VOLUME-SPLIT",
                "severity": "medium",
                "issue_type": "split_volume_convention",
                "status": "unresolved",
                "evidence": (
                    f"{len(split):,} split events have volume-neighborhood metrics; "
                    "Yahoo does not document a single retrospective share-volume "
                    "normalization convention in these files."
                ),
                "remaining_uncertainty": (
                    "Event and neighborhood effects mix trading activity, reverse "
                    "split mechanics, vendor corrections, and possible volume "
                    "normalization."
                ),
                "recommended_action": (
                    "Retain reported share volume unchanged. Do not compare "
                    "pre/post-split share volume until a future documented "
                    "normalization rule is applied."
                ),
            },
        ]
    )
    _write_csv(issues, output_dir / "volume_issues.csv")
    _figures(year, security, split, figure_dir)

    master = pd.read_parquet(master_path)
    stats = {
        "provider_counts": raw["provider"].value_counts().to_dict(),
        "rows": int(len(raw)),
        **concentration,
        "zero_volume_rows": int(security["zero_volume_rows"].sum()),
        "securities_with_any_zero": int(
            security["zero_volume_rows"].gt(0).sum()
        ),
        "securities_zero_rate_ge_0_5": int(
            security["zero_volume_rate"].ge(0.5).sum()
        ),
        "prolonged_cases": int(len(prolonged)),
        "split_events_with_volume_metrics": int(len(split)),
        "split_events_with_positive_pre5_median": int(
            split["pre5_median_volume"].gt(0).sum()
        ),
        "master_securities": int(len(master)),
    }
    atomic_write_json(stats, output_dir / "volume_investigation_summary.json")
    (PROJECT_ROOT / "docs" / "investigation_volume.md").write_text(
        f"""# Volume-quality investigation

The validated Yahoo panel contains {stats['zero_volume_rows']:,} zero-volume
rows across {stats['securities_with_any_zero']:,} securities. Half of these
rows are concentrated in {stats['securities_for_first_half']:,} securities.
{stats['prolonged_cases']:,} securities meet the descriptive prolonged-case
screen of at least 50% zero-volume observations or a contiguous run of at
least 252 rows.

Zero volume is retained as zero, not converted to missing. It may represent
illiquidity, a suspension, an inactive vendor row, or another unresolved
condition; this audit does not infer which without additional evidence.

The panel contains {stats['split_events_with_volume_metrics']:,} split events
with prior-session volume. Reported share volume is retained without
normalization. The event neighborhood mixes genuine trading, split mechanics,
and vendor conventions, so pre/post-split volume must not be interpreted as
constant-share activity without a later documented transformation.
""",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "acquisition.yaml")
    )
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
