"""Create the final v3 research panel by removing verified identity failure B.

All inputs are read-only. The output follows the artifact conventions of the
eligibility-corrected v2 dataset while adding separate security-level exclusion
and removed-observation audits.
"""

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


LOGGER = configure_logging(PROJECT_ROOT / "logs" / "final_dataset_v3.log")


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


def _update_quality_summary(
    parent_summary: pd.DataFrame,
    *,
    parent_securities: int,
    final_securities: int,
    parent_issuers: int,
    final_issuers: int,
    parent_rows: int,
    final_rows: int,
    removed_rows: int,
    removed_ticker: str,
) -> pd.DataFrame:
    summary = parent_summary.copy()
    replacements = {
        "Securities": (parent_securities, final_securities),
        "Distinct SEC issuers": (parent_issuers, final_issuers),
        "Validated stock-day observations": (parent_rows, final_rows),
    }
    for metric, (_, final_value) in replacements.items():
        mask = summary["metric"].eq(metric)
        summary.loc[mask, "final_research_dataset"] = final_value

    insert_at = int(
        summary.index[summary["metric"].eq("DKNG observations excluded")][0]
    ) + 1
    removal_row = pd.DataFrame(
        [
            {
                "metric": f"{removed_ticker} security-removal observations",
                "original_acquired": 0,
                "original_liquidity_filtered": 0,
                "final_research_dataset": removed_rows,
                "original_acquired_per_100k": np.nan,
                "original_liquidity_filtered_per_100k": np.nan,
                "final_research_dataset_per_100k": np.nan,
            }
        ]
    )
    summary = pd.concat(
        [
            summary.iloc[:insert_at],
            removal_row,
            summary.iloc[insert_at:],
        ],
        ignore_index=True,
    )

    event_metrics = [
        "Zero-volume observations",
        "Extreme-volume observations",
        "Extreme diagnostic-return observations",
        "Nonpositive adjusted-price observations",
        "Existing price-level case records",
        "Internal missing XNYS sessions",
        "Unresolved extreme-return discontinuities",
        "Corporate-action discrepancy records",
        "Quarantine-candidate rows/windows",
    ]
    denominator = final_rows
    for metric in event_metrics:
        mask = summary["metric"].eq(metric)
        value = float(summary.loc[mask, "final_research_dataset"].iloc[0])
        summary.loc[mask, "final_research_dataset_per_100k"] = (
            100_000.0 * value / denominator if denominator else np.nan
        )
    return summary


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(project_path(config_path))
    inputs = config["inputs"]
    parent_root = project_path(inputs["parent_root"])
    audit_root = project_path(inputs["identity_audit_root"])
    audit_summary_path = project_path(inputs["identity_audit_summary"])
    audit_table_path = project_path(inputs["security_identity_audit"])
    manifest_path = project_path(inputs["download_manifest"])
    output_root = project_path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)

    protected_paths = [
        parent_root / "selected_equity_daily.parquet",
        parent_root / "selected_security_master.parquet",
        parent_root / "diagnostic_returns.parquet",
        parent_root / "historical_exclusions.csv",
        parent_root / "final_data_quality_summary.csv",
        parent_root / "cleaning_policy.json",
        audit_summary_path,
        audit_table_path,
        manifest_path,
    ]
    before_hashes = {str(path): _sha256(path) for path in protected_paths}
    before_parent_tree = tree_sha256(parent_root)
    before_audit_tree = tree_sha256(audit_root)

    master = pd.read_parquet(parent_root / "selected_security_master.parquet")
    daily = pd.read_parquet(parent_root / "selected_equity_daily.parquet")
    diagnostics = pd.read_parquet(parent_root / "diagnostic_returns.parquet")
    coverage = pd.read_csv(parent_root / "security_coverage.csv", low_memory=False)
    zero_cases = pd.read_csv(
        parent_root / "remaining_zero_volume_cases.csv", low_memory=False
    )
    zero_by_security = pd.read_csv(
        parent_root / "remaining_zero_volume_by_security.csv", low_memory=False
    )
    extremes = pd.read_csv(
        parent_root / "remaining_extreme_value_cases.csv", low_memory=False
    )
    historical_exclusions = pd.read_csv(
        parent_root / "historical_exclusions.csv", low_memory=False
    )
    parent_summary = pd.read_csv(
        parent_root / "final_data_quality_summary.csv", low_memory=False
    )
    parent_cleaning = json.loads(
        (parent_root / "cleaning_policy.json").read_text(encoding="utf-8")
    )
    audit_summary = json.loads(
        audit_summary_path.read_text(encoding="utf-8")
    )
    audit_table = pd.read_csv(
        audit_table_path, dtype=str, low_memory=False, keep_default_na=False
    )
    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest["security_id"] = manifest["security_id"].astype(str)
    security_ids = [str(value) for value in config["exclusions"]["security_ids"]]
    if len(security_ids) != 1:
        raise RuntimeError("This v3 export expects exactly one security exclusion")
    excluded_id = security_ids[0]
    if daily.security_id.astype(str).eq(excluded_id).sum() == 0:
        raise RuntimeError(f"Excluded security is absent from v2: {excluded_id}")
    identity_row = audit_table[audit_table["security_id"].eq(excluded_id)]
    if len(identity_row) != 1:
        raise RuntimeError("Excluded security lacks a unique identity-audit row")
    if identity_row["identity_audit_status"].iloc[0] != "vendor_series_mismatch":
        raise RuntimeError("Excluded security is not audited as vendor_series_mismatch")
    expected_ticker = identity_row["ticker_2023"].iloc[0]

    excluded_daily = daily[daily.security_id.astype(str).eq(excluded_id)].copy()
    final_daily = daily[~daily.security_id.astype(str).eq(excluded_id)].copy()
    final_master = master[~master.security_id.astype(str).eq(excluded_id)].copy()
    final_diagnostics = diagnostics[
        ~diagnostics.security_id.astype(str).eq(excluded_id)
    ].copy()
    final_coverage = coverage[~coverage.security_id.astype(str).eq(excluded_id)].copy()
    final_zero = zero_cases[
        ~zero_cases.security_id.astype(str).eq(excluded_id)
    ].copy()
    final_zero_by_security = zero_by_security[
        ~zero_by_security.security_id.astype(str).eq(excluded_id)
    ].copy()
    final_extremes = extremes[
        ~extremes.security_id.astype(str).eq(excluded_id)
    ].copy()

    if len(final_master) != 487 or final_master["cik"].nunique() != 485:
        raise RuntimeError("Unexpected v3 security or issuer count")
    if len(final_daily) != 1_183_682:
        raise RuntimeError("Unexpected v3 validated row count")
    if final_daily.duplicated(["security_id", "date"]).any():
        raise RuntimeError("v3 contains duplicate security-date keys")
    if not final_daily["pct_return"].isna().all():
        raise RuntimeError("v3 pct_return must remain null")
    removed_flags = diagnostics[
        diagnostics.security_id.astype(str).eq(excluded_id)
    ]
    surprising_flags = any(
        int(removed_flags[column].fillna(False).sum()) > 0
        for column in (
            "zero_volume_flag",
            "extreme_volume_spike_flag",
            "extreme_return_flag",
            "zero_or_negative_adj_close_flag",
        )
    )
    if surprising_flags:
        raise RuntimeError(
            "Excluded security has anomaly flags; update quality counts explicitly"
        )

    excluded_manifest = manifest[
        manifest["security_id"].eq(excluded_id)
    ].sort_values("completed_at_utc")
    source_file = (
        str(excluded_manifest["raw_path"].dropna().iloc[-1])
        if excluded_manifest["raw_path"].notna().any()
        else ""
    )
    decision = {
        "security_id": excluded_id,
        "cik": identity_row["cik"].iloc[0],
        "ticker": expected_ticker,
        "company_name": identity_row["company_name"].iloc[0],
        "exchange": identity_row["exchange_2023"].iloc[0],
        "exclusion_reason": config["exclusions"]["reason"],
        "policy_type": config["exclusions"]["policy_type"],
        "source_reference": config["exclusions"]["source_reference"],
        "verification_status": identity_row["identity_audit_status"].iloc[0],
        "decision_date": config["project"]["decision_date"],
        "observations_removed": int(len(excluded_daily)),
        "first_observation": excluded_daily["date"].min().date().isoformat(),
        "last_observation": excluded_daily["date"].max().date().isoformat(),
        "original_source_file": source_file,
    }
    security_exclusions = pd.DataFrame([decision])
    removed_observations = excluded_daily.assign(
        exclusion_reason=decision["exclusion_reason"],
        policy_reference=decision["source_reference"],
        original_source_file=source_file,
    )

    final_summary = _update_quality_summary(
        parent_summary,
        parent_securities=len(master),
        final_securities=len(final_master),
        parent_issuers=int(master["cik"].nunique()),
        final_issuers=int(final_master["cik"].nunique()),
        parent_rows=len(daily),
        final_rows=len(final_daily),
        removed_rows=len(excluded_daily),
        removed_ticker=expected_ticker,
    )

    snapshot = dict(parent_cleaning["snapshot"])
    snapshot.update(
        {
            "securities": int(len(final_master)),
            "issuers": int(final_master["cik"].nunique()),
            "observations": int(len(final_daily)),
            "diagnostic_rows": int(len(final_diagnostics)),
            "security_identity_exclusions": 1,
            "security_identity_exclusion_rows": int(len(excluded_daily)),
            "fixed_membership_unchanged": False,
            "membership_changed_from_v2": True,
            "parent_v2_securities": int(len(master)),
        }
    )
    policy = dict(parent_cleaning)
    policy.update(
        {
            "generated_at_utc": utc_now(),
            "version": config["project"]["name"],
            "parent_version": config["project"]["parent_version"],
            "snapshot": snapshot,
            "security_exclusion_policy": [decision],
            "preserved_parent_policy": {
                "historical_eligibility": parent_cleaning.get("policy", []),
                "treatments": parent_cleaning.get("treatments", []),
            },
        }
    )
    policy["treatments"] = [
        {
            "issue": f"{expected_ticker} Yahoo checkpoint identity mismatch",
            "final_treatment": "exclude_security_from_final_research_dataset",
        },
        *parent_cleaning.get("treatments", []),
    ]
    policy["protected_parent_hashes_sha256"] = before_hashes
    policy["software"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }

    _atomic_parquet(final_daily, output_root / "selected_equity_daily.parquet")
    _atomic_parquet(final_master, output_root / "selected_security_master.parquet")
    _atomic_parquet(
        final_diagnostics, output_root / "diagnostic_returns.parquet"
    )
    _atomic_csv(historical_exclusions, output_root / "historical_exclusions.csv")
    _atomic_csv(security_exclusions, output_root / "security_exclusions.csv")
    _atomic_csv(
        removed_observations,
        output_root / "removed_security_observations.csv",
    )
    _atomic_csv(final_summary, output_root / "final_data_quality_summary.csv")
    _atomic_csv(final_zero, output_root / "remaining_zero_volume_cases.csv")
    _atomic_csv(
        final_zero_by_security,
        output_root / "remaining_zero_volume_by_security.csv",
    )
    _atomic_csv(
        final_extremes, output_root / "remaining_extreme_value_cases.csv"
    )
    _atomic_csv(final_coverage, output_root / "security_coverage.csv")
    atomic_write_json(policy, output_root / "cleaning_policy.json")

    execution = {
        "status": "complete",
        "generated_at_utc": utc_now(),
        "snapshot": snapshot,
        "security_exclusions": security_exclusions.to_dict("records"),
        "preserved_historical_exclusion_rows": int(len(historical_exclusions)),
        "protected_inputs_unchanged": True,
    }
    atomic_write_json(execution, output_root / "execution_summary.json")
    (output_root / "README.md").write_text(
        f"""# Final research dataset v3

This directory extends `adtv100m_2023_eligibility_v2` by removing one security
whose Yahoo checkpoint was proven to represent a different issuer.

- Removed: `{expected_ticker}` ({decision['company_name']}, `{excluded_id}`)
- Removed validated observations: {len(excluded_daily):,}
- Final securities: {len(final_master):,}
- Final distinct SEC issuers: {final_master['cik'].nunique():,}
- Final validated observations: {len(final_daily):,}

The reason for exclusion is the verified Barnes/Barrick identity mismatch
documented in `security_exclusions.csv` and the 488-security identity audit.
The v2 FERG and DKNG historical date exclusions are preserved. No source price,
volume, diagnostic, or quality record was changed.
""",
        encoding="utf-8",
    )

    after_hashes = {str(path): _sha256(path) for path in protected_paths}
    if (
        before_hashes != after_hashes
        or before_parent_tree != tree_sha256(parent_root)
        or before_audit_tree != tree_sha256(audit_root)
    ):
        raise RuntimeError("A protected v2 or identity-audit input changed")
    execution["protected_inputs_unchanged"] = True
    atomic_write_json(execution, output_root / "execution_summary.json")
    LOGGER.info(
        "Built v3 with %d securities and %d observations",
        len(final_master),
        len(final_daily),
    )
    return execution


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "final_dataset_v3.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
