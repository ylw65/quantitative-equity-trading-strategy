"""Fail when report statistics, generated prose, and source manifests disagree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import PROJECT_ROOT, atomic_write_json


LEGACY_ABSTRACT_PHRASES = (
    "all \\YahooAttempted{} representative attempts failed",
    "secondary public-demo sample provides",
    "primary Yahoo Finance request path was unavailable",
)


def check_report_consistency(
    root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    stats = json.loads(
        (root / "reports/latex/generated/report_statistics.json").read_text()
    )
    manifest = pd.read_csv(
        root / "data/metadata/download_manifest.csv", low_memory=False
    )
    base = pd.read_parquet(
        root / "data/validated/base.parquet",
        columns=["security_id", "date", "pct_return"],
    )
    abstract_path = root / "reports/latex/generated/abstract.tex"
    main_tex = (root / "reports/latex/main.tex").read_text()
    abstract = abstract_path.read_text() if abstract_path.exists() else ""
    provenance_path = (
        root / "data/investigation/identity/provenance_reconciliation.json"
    )
    provenance = (
        json.loads(provenance_path.read_text())
        if provenance_path.exists()
        else {}
    )
    validation_provenance = json.loads(
        (root / "data/validated/validation_provenance.json").read_text()
    )
    body_files = [
        root / "reports/latex/generated/executive_summary.tex",
        root / "reports/latex/generated/acquisition_results.tex",
        root / "reports/latex/generated/coverage_results.tex",
        root / "reports/latex/generated/conclusions.tex",
    ]
    body = "\n".join(path.read_text() for path in body_files if path.exists())

    checks = [
        (
            "manifest_attempted_matches_statistics",
            len(manifest),
            stats["acquisition"]["yahoo_attempted"],
        ),
        (
            "manifest_success_matches_statistics",
            int(manifest["download_status"].eq("success").sum()),
            stats["acquisition"]["yahoo_success"],
        ),
        (
            "manifest_failed_matches_statistics",
            int(manifest["download_status"].eq("failed").sum()),
            stats["acquisition"]["yahoo_failed"],
        ),
        (
            "base_rows_match_statistics",
            len(base),
            stats["data"]["observations"],
        ),
        (
            "base_security_count_matches_statistics",
            int(base["security_id"].nunique()),
            stats["data"]["securities"],
        ),
        (
            "base_pct_return_null",
            bool(base["pct_return"].isna().all()),
            True,
        ),
        (
            "provider_success_reconciles_investigation",
            provenance.get("manifest_success"),
            stats["acquisition"]["yahoo_success"],
        ),
        (
            "provider_raw_rows_reconcile_investigation",
            provenance.get("raw_observation_count_from_manifest"),
            validation_provenance.get("diagnostic_rows"),
        ),
        (
            "standardized_raw_rows_reconcile_investigation",
            provenance.get("validation_standardized_rows"),
            validation_provenance.get("standardized_raw_rows"),
        ),
        (
            "eodhd_outside_yahoo_raw",
            provenance.get("eod_demo_paths_inside_yahoo_raw"),
            [],
        ),
        (
            "abstract_artifact_exists",
            abstract_path.exists(),
            True,
        ),
        (
            "main_tex_uses_generated_abstract",
            "\\InputGenerated{generated/abstract.tex}" in main_tex,
            True,
        ),
        (
            "abstract_uses_current_success_macro",
            "\\YahooSuccess{}" in abstract,
            True,
        ),
        (
            "abstract_uses_current_observation_macro",
            "\\Observations{}" in abstract,
            True,
        ),
        (
            "body_uses_current_success_macro",
            "\\YahooSuccess{}" in body,
            True,
        ),
        (
            "legacy_abstract_phrases_absent",
            not any(phrase in abstract for phrase in LEGACY_ABSTRACT_PHRASES),
            True,
        ),
    ]
    rows = []
    for name, actual, expected in checks:
        rows.append(
            {
                "check": name,
                "actual": actual,
                "expected": expected,
                "passed": actual == expected,
            }
        )
    frame = pd.DataFrame(rows)
    payload = {
        "passed": bool(frame["passed"].all()),
        "checks_total": len(frame),
        "checks_passed": int(frame["passed"].sum()),
        "checks_failed": int((~frame["passed"]).sum()),
        "statistics_generated_at_utc": stats.get("generated_at_utc"),
        "source_manifests": {
            "download_manifest": "data/metadata/download_manifest.csv",
            "validation_provenance": "data/validated/validation_provenance.json",
            "identity_reconciliation": "data/investigation/identity/provenance_reconciliation.json",
        },
    }
    return payload, frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    payload, checks = check_report_consistency(args.root)
    output_dir = args.root / "data/investigation/report_consistency"
    output_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(output_dir / "checks.csv", index=False)
    atomic_write_json(payload, output_dir / "summary.json")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
