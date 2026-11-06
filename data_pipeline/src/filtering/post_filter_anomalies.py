"""Post-filter anomaly, volume, and corporate-action assessment."""

from __future__ import annotations

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

from src.common import PROJECT_ROOT, atomic_write_json, load_config, project_path


COLORS = {
    "navy": "#19324D",
    "blue": "#3F6F8F",
    "teal": "#4E8C86",
    "gold": "#B28A48",
    "rust": "#A45D4F",
    "slate": "#65727A",
}


def selected_id_set(master: pd.DataFrame) -> set[str]:
    """Return the filtered security set from the selected master."""
    return set(master["security_id"].dropna().astype(str))


def diagnostic_metrics(
    diagnostics: pd.DataFrame, security_ids: set[str]
) -> dict[str, int | float]:
    """Calculate diagnostic-return and volume flags for one cohort."""
    frame = diagnostics.loc[diagnostics["security_id"].astype(str).isin(security_ids)]
    extreme = frame["extreme_return_flag"].fillna(False).astype(bool)
    volume_spike = frame["extreme_volume_spike_flag"].fillna(False).astype(bool)
    zero_volume = frame["zero_volume_flag"].fillna(False).astype(bool)
    return {
        "rows": int(len(frame)),
        "securities": int(frame["security_id"].nunique()),
        "extreme_returns": int(extreme.sum()),
        "extreme_return_securities": int(
            frame.loc[extreme, "security_id"].nunique()
        ),
        "extreme_return_rate": float(extreme.mean()) if len(frame) else np.nan,
        "extreme_volume_spikes": int(volume_spike.sum()),
        "extreme_volume_spike_securities": int(
            frame.loc[volume_spike, "security_id"].nunique()
        ),
        "extreme_volume_spike_rate": (
            float(volume_spike.mean()) if len(frame) else np.nan
        ),
        "zero_volume_rows": int(zero_volume.sum()),
        "zero_volume_securities": int(
            frame.loc[zero_volume, "security_id"].nunique()
        ),
        "zero_volume_rate": (
            float(zero_volume.mean()) if len(frame) else np.nan
        ),
    }


def filter_artifact(
    frame: pd.DataFrame, security_ids: set[str]
) -> pd.DataFrame:
    """Filter an evidence artifact by selected security ID."""
    return frame.loc[frame["security_id"].astype(str).isin(security_ids)].copy()


def action_metrics(
    selected_quality: pd.DataFrame,
    selected_issues: pd.DataFrame,
    split_reconciliation: pd.DataFrame,
    dividend_reconciliation: pd.DataFrame,
) -> dict[str, int]:
    """Calculate corporate-action counts from selected-cohort evidence."""
    return {
        "split_events": int(selected_quality["split_events"].fillna(0).sum()),
        "dividend_events": int(
            selected_quality["dividend_events"].fillna(0).sum()
        ),
        "split_continuity_flags": int(
            selected_quality["split_continuity_flags"].fillna(0).sum()
        ),
        "dividend_continuity_flags": int(
            selected_quality["dividend_continuity_flags"].fillna(0).sum()
        ),
        "simultaneous_split_distribution_groups": int(
            selected_issues["issue_type"]
            .eq("simultaneous_split_and_distribution")
            .sum()
        ),
        "simultaneous_split_distribution_events": int(
            pd.to_numeric(
                selected_issues.loc[
                    selected_issues["issue_type"].eq(
                        "simultaneous_split_and_distribution"
                    ),
                    "affected_rows",
                ],
                errors="coerce",
            ).fillna(0).sum()
        ),
        "split_reconciliation_events": int(len(split_reconciliation)),
        "dividend_reconciliation_events": int(len(dividend_reconciliation)),
        "ordinary_dividend_unresolved": int(
            dividend_reconciliation["final_classification"]
            .eq("ordinary_dividend_adjustment_mismatch")
            .sum()
        ),
    }


def _classification_table(
    frame: pd.DataFrame, column: str, label: str
) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return pd.DataFrame(columns=[label, "rows"])
    return (
        frame[column]
        .fillna("unknown")
        .value_counts()
        .rename_axis(label)
        .reset_index(name="rows")
    )


def _save_figures(
    output_dir: Path,
    extreme_classes: pd.DataFrame,
    before_after: pd.DataFrame,
    action_classes: pd.DataFrame,
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
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    if not extreme_classes.empty:
        values = extreme_classes.sort_values("rows")
        ax.barh(
            values.iloc[:, 0].astype(str).str.replace("_", " "),
            values["rows"],
            color=COLORS["blue"],
        )
        ax.set_xlabel("Rows or windows")
    ax.set_title("Extreme observations in ADTV-filtered cohort")
    fig.savefig(output_dir / "postfilter_extreme_categories.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    metrics = before_after.set_index("metric")
    rates = metrics.loc[
        ["zero-volume rate", "extreme-return rate", "extreme-volume-spike rate"]
    ]
    x = np.arange(len(rates))
    width = 0.36
    ax.bar(
        x - width / 2,
        rates["original_rate"].astype(float) * 100,
        width,
        color=COLORS["slate"],
        label="original",
    )
    ax.bar(
        x + width / 2,
        rates["filtered_rate"].astype(float) * 100,
        width,
        color=COLORS["teal"],
        label="filtered",
    )
    ax.set_xticks(x, [value.replace(" rate", "") for value in rates.index], rotation=15)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Quality-diagnostic rates before and after selection")
    ax.legend()
    fig.savefig(output_dir / "postfilter_quality_rates.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    if not action_classes.empty:
        values = action_classes.sort_values("events")
        ax.barh(
            values["classification"].str.replace("_", " "),
            values["events"],
            color=COLORS["gold"],
        )
        ax.set_xlabel("Events")
    ax.set_title("Reconciled corporate actions in filtered cohort")
    fig.savefig(output_dir / "postfilter_corporate_actions.pdf")
    plt.close(fig)


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(project_path(config_path))
    input_cfg = config["inputs"]
    output_root = project_path(config["outputs"]["root"])
    quality_dir = output_root / "quality"
    table_dir = project_path(config["outputs"]["tables"])
    figure_dir = project_path(config["outputs"]["figures"])
    quality_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    selected_master = pd.read_parquet(
        output_root / "selected_security_master.parquet"
    )
    selected_daily_rows = len(
        pd.read_parquet(
            output_root / "selected_equity_daily.parquet",
            columns=["security_id"],
        )
    )
    ids = selected_id_set(selected_master)
    quality = pd.read_csv(
        project_path(input_cfg["quality_by_security"]), low_memory=False
    )
    issues = pd.read_csv(
        project_path(input_cfg["validation_issues"]), low_memory=False
    )
    diagnostics = pd.read_parquet(
        project_path(input_cfg["diagnostic_returns"]),
        columns=[
            "date",
            "security_id",
            "ticker",
            "diagnostic_pct_return",
            "extreme_return_flag",
            "extreme_volume_spike_flag",
            "zero_volume_flag",
        ],
    )
    anomaly_dir = PROJECT_ROOT / "data/investigation/anomalies"
    volume_dir = PROJECT_ROOT / "data/investigation/volume"
    extreme_cases = pd.read_csv(
        anomaly_dir / "extreme_return_cases.csv", low_memory=False
    )
    price_cases = pd.read_csv(
        anomaly_dir / "extreme_price_cases.csv", low_memory=False
    )
    quarantine = pd.read_csv(
        anomaly_dir / "analysis_quarantine_candidates.csv", low_memory=False
    )
    anomaly_issues = pd.read_csv(
        anomaly_dir / "anomaly_issues.csv", low_memory=False
    )
    split_recon = pd.read_csv(
        anomaly_dir / "corporate_action_split_reconciliation.csv",
        low_memory=False,
    )
    dividend_recon = pd.read_csv(
        anomaly_dir / "corporate_action_dividend_reconciliation.csv",
        low_memory=False,
    )
    volume_security = pd.read_csv(
        volume_dir / "zero_volume_by_security.csv", low_memory=False
    )
    identity_last_trade = pd.read_csv(
        PROJECT_ROOT / "data/investigation/identity/security_last_trade.csv",
        low_memory=False,
    )

    selected_quality = quality.loc[quality["security_id"].isin(ids)].copy()
    selected_issues = issues.loc[issues["security_id"].isin(ids)].copy()
    selected_diagnostics = filter_artifact(diagnostics, ids)
    selected_extremes = filter_artifact(extreme_cases, ids)
    selected_prices = filter_artifact(price_cases, ids)
    selected_quarantine = filter_artifact(quarantine, ids)
    selected_anomaly_issues = filter_artifact(anomaly_issues, ids)
    selected_split_recon = filter_artifact(split_recon, ids)
    selected_dividend_recon = filter_artifact(dividend_recon, ids)
    selected_volume = filter_artifact(volume_security, ids)
    selected_trailing = filter_artifact(identity_last_trade, ids)
    if "stopped_trading_candidate" in selected_trailing:
        trailing_zero = selected_trailing[
            selected_trailing["stopped_trading_candidate"].fillna(False)
        ]
    else:
        trailing_zero = selected_trailing.iloc[0:0]

    original_metrics = diagnostic_metrics(
        diagnostics, set(diagnostics["security_id"].astype(str))
    )
    filtered_metrics = diagnostic_metrics(diagnostics, ids)
    actions = action_metrics(
        selected_quality,
        selected_issues,
        selected_split_recon,
        selected_dividend_recon,
    )

    extreme_classes = _classification_table(
        selected_extremes, "classification", "classification"
    )
    selected_extremes.to_csv(
        quality_dir / "filtered_extreme_return_cases.csv", index=False
    )
    selected_prices.to_csv(
        quality_dir / "filtered_extreme_price_cases.csv", index=False
    )
    selected_quarantine.to_csv(
        quality_dir / "filtered_quarantine_candidates.csv", index=False
    )
    selected_anomaly_issues.to_csv(
        quality_dir / "filtered_anomaly_issues.csv", index=False
    )
    selected_split_recon.to_csv(
        quality_dir / "filtered_split_reconciliation.csv", index=False
    )
    selected_dividend_recon.to_csv(
        quality_dir / "filtered_dividend_reconciliation.csv", index=False
    )
    selected_volume.to_csv(
        quality_dir / "filtered_volume_by_security.csv", index=False
    )
    extreme_classes.to_csv(
        quality_dir / "filtered_extreme_return_classes.csv", index=False
    )

    combined_action_classes = pd.concat(
        [
            selected_split_recon.assign(
                classification=selected_split_recon["final_classification"]
            )[["classification"]],
            selected_dividend_recon.assign(
                classification=selected_dividend_recon["final_classification"]
            )[["classification"]],
        ],
        ignore_index=True,
    )
    action_classes = _classification_table(
        combined_action_classes, "classification", "classification"
    ).rename(columns={"rows": "events"})

    before_after = pd.DataFrame(
        [
            {
                "metric": "zero-volume rate",
                "original_count": original_metrics["zero_volume_rows"],
                "filtered_count": filtered_metrics["zero_volume_rows"],
                "original_rate": original_metrics["zero_volume_rate"],
                "filtered_rate": filtered_metrics["zero_volume_rate"],
            },
            {
                "metric": "extreme-return rate",
                "original_count": original_metrics["extreme_returns"],
                "filtered_count": filtered_metrics["extreme_returns"],
                "original_rate": original_metrics["extreme_return_rate"],
                "filtered_rate": filtered_metrics["extreme_return_rate"],
            },
            {
                "metric": "extreme-volume-spike rate",
                "original_count": original_metrics["extreme_volume_spikes"],
                "filtered_count": filtered_metrics["extreme_volume_spikes"],
                "original_rate": original_metrics["extreme_volume_spike_rate"],
                "filtered_rate": filtered_metrics["extreme_volume_spike_rate"],
            },
        ]
    )
    before_after.to_csv(
        quality_dir / "before_after_anomaly_volume.csv", index=False
    )
    before_after.to_csv(
        table_dir / "postfilter_before_after_anomaly_volume.csv", index=False
    )
    extreme_classes.to_csv(
        table_dir / "postfilter_extreme_return_classes.csv", index=False
    )
    action_classes.to_csv(
        table_dir / "postfilter_corporate_action_classes.csv", index=False
    )

    summary = {
        "selected_securities": len(ids),
        "selected_validated_rows": selected_daily_rows,
        "selected_diagnostic_rows": int(len(selected_diagnostics)),
        "original_diagnostic_metrics": original_metrics,
        "filtered_diagnostic_metrics": filtered_metrics,
        "extreme_return_classifications": {
            row["classification"]: int(row["rows"])
            for _, row in extreme_classes.iterrows()
        },
        "extreme_price_securities": int(
            selected_prices["security_id"].nunique()
        ),
        "quarantine_rows_windows": int(len(selected_quarantine)),
        "quarantine_securities": int(
            selected_quarantine["security_id"].nunique()
        ),
        "zero_volume_securities": int(
            selected_volume.loc[
                selected_volume["zero_volume_rows"].gt(0), "security_id"
            ].nunique()
        ),
        "prolonged_zero_volume_securities": int(
            (
                selected_volume["zero_volume_rate"].ge(0.5)
                | selected_volume["longest_zero_run_rows"].ge(252)
            ).sum()
        ),
        "trailing_zero_volume_candidates": int(len(trailing_zero)),
        "corporate_actions": actions,
        "corporate_action_classes": {
            row["classification"]: int(row["events"])
            for _, row in action_classes.iterrows()
        },
        "source_data_modified": False,
    }
    atomic_write_json(summary, quality_dir / "post_filter_anomaly_summary.json")
    _save_figures(figure_dir, extreme_classes, before_after, action_classes)

    (PROJECT_ROOT / "docs" / "post_filter_anomalies.md").write_text(
        f"""# Post-filter anomaly, volume, and corporate-action assessment

The ADTV-filtered cohort contains {len(ids):,} securities and
{filtered_metrics['rows']:,} diagnostic rows. It contains
{filtered_metrics['extreme_returns']:,} diagnostic returns beyond the absolute
50 percent threshold across {filtered_metrics['extreme_return_securities']:,}
securities. Near-zero-denominator and other evidence classifications are
recomputed in the filtered files rather than copied from the original report.

The cohort has {filtered_metrics['zero_volume_rows']:,} zero-volume rows across
{summary['zero_volume_securities']:,} securities and
{summary['prolonged_zero_volume_securities']:,} prolonged-zero-volume cases.
The filtered data also contain {filtered_metrics['extreme_volume_spikes']:,}
extreme-volume-spike flags.

Corporate-action evidence is filtered to the same security IDs. The cohort has
{actions['split_events']:,} split events, {actions['dividend_events']:,}
dividend/distribution events, {actions['split_continuity_flags']:,} simple
split-continuity flags, {actions['dividend_continuity_flags']:,} simple
dividend-continuity flags, and
{actions['simultaneous_split_distribution_events']:,} simultaneous
split/distribution events.

The analysis-quarantine register overlaps
{summary['quarantine_rows_windows']:,} rows/windows across
{summary['quarantine_securities']:,} selected securities. All raw data and the
original research base remain unchanged.
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "liquidity_filter.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
