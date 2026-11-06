from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common import PROJECT_ROOT
from src.eda.eda import GEN_DIR, STATS_PATH, TABLE_DIR, _tex


pytestmark = pytest.mark.skipif(
    not STATS_PATH.exists(), reason="EDA artifacts have not been generated"
)


def test_eda_statistics_match_primary_artifacts() -> None:
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    master = pd.read_parquet(PROJECT_ROOT / "data/universe/security_master.parquet")
    base = pd.read_parquet(PROJECT_ROOT / "data/validated/base.parquet")
    universe_issues = pd.read_csv(
        PROJECT_ROOT / "data/universe/universe_issues.csv", low_memory=False
    )
    assert stats["universe"]["included_securities"] == len(master)
    assert stats["universe"]["issue_groups"] == len(universe_issues)
    assert stats["data"]["observations"] == len(base)
    assert stats["data"]["securities"] == base["security_id"].nunique()
    assert stats["auditability"]["duplicate_base_keys"] == 0
    assert base["pct_return"].isna().all()


def test_generated_report_tables_and_figures_are_present() -> None:
    required_tables = {
        "universe_by_exchange.csv",
        "universe_by_type.csv",
        "download_status.csv",
        "coverage_by_year.csv",
        "missingness_by_year.csv",
        "split_audit.csv",
        "dividend_audit.csv",
        "issue_summary.csv",
    }
    required_figures = {
        "universe_composition.pdf",
        "download_status.pdf",
        "observations_by_year.pdf",
        "representative_histories.pdf",
        "adjusted_close_distribution.pdf",
        "volume_distribution_log.pdf",
        "split_adjustment_audit.pdf",
        "diagnostic_returns_tail.pdf",
    }
    assert required_tables.issubset({path.name for path in TABLE_DIR.glob("*.csv")})
    assert required_figures.issubset(
        {path.name for path in (PROJECT_ROOT / "reports/figures").glob("*.pdf")}
    )
    generated_tex = "\n".join(
        path.read_text(encoding="utf-8") for path in GEN_DIR.glob("*.tex")
    )
    assert "Empirical artifact pending" not in generated_tex


def test_latex_escape_formats_numbers_without_double_escaping() -> None:
    assert _tex(4874) == "4,874"
    assert _tex(2023) == "2023"
    assert _tex("35.5%") == r"35.5\%"
    assert "textbackslash" not in _tex("common_stock")
