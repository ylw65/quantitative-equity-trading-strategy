# Historical US Equity Data Acquisition and Quality Assessment

> Data pipeline source: upstream `main` snapshot.
> Historical counts and generated report tables below are archived research
> records, not a new execution. Raw and intermediate data are not bundled.
> The top-level [reproduction guide](../docs/REPRODUCIBILITY.md) explains the
> separate environment, portable tests and final handoff to the strategy.
> Artifact-dependent test assertions require `--run-artifact-tests` and the
> complete original workspace. Default skips are reported explicitly.

This project reconstructs a best-verifiable reference-date US common-equity
universe, implements a resumable Yahoo Finance acquisition pipeline, validates
daily market observations, generates exploratory diagnostics, and produces a
reproducible LaTeX/PDF quality-assessment report.

The phase is limited to acquisition, standardization, validation, EDA, and
documentation. It does not create final returns, implement train/validation/test
splits, construct strategy features, train models, or run backtests.

## Current result

- Reference date: 3 January 2023.
- Verified partial universe: 4,874 unique common-equity securities.
- SEC cover rows audited: 9,047.
- Exchange composition: 3,038 Nasdaq, 1,729 NYSE, 107 NYSE American.
- Universe issue groups: 6,554.
- Primary Yahoo universe: 4,874 attempted; 3,239 successful histories and
  1,635 unavailable/empty symbols.
- Secondary EODHD public-demo sample: 40 requested; 3 successful (AAPL, MSFT,
  TSLA), 37 restricted/failed.
- Raw Yahoo observations: 7,128,162; validated research base: 7,086,540 rows
  from 2016-01-04 through 2025-12-31.
- Final research-facing `pct_return`: null for every row.
- Checkpoint identity audit: 3,239/3,239 valid; no duplicate date keys or exact
  cross-security histories.
- Report consistency: 16/16 checks pass against the manifest, validation
  provenance, provider labels, base identifiers, and generated abstract.
- Fixed liquidity cohort: 488 securities and 486 distinct SEC issuers pass the
  inclusive USD 100 million ADTV20 screen over the final 20 XNYS sessions of
  2022. The selected full-history panel has 1,187,986 validated stock-day rows.
- Liquidity-report consistency: 39/39 checks pass against source hashes,
  selection counts, selected row counts, generated tables/macros, and
  post-filter audit summaries.
- Filtered anomaly-episode extraction: 533 strict XNYS-session episodes across
  134 securities, covering 1,212 flagged stock-days. All 1,207 diagnostic-flag
  rows are assigned exactly once, while four corporate-action reconciliation
  records and two quarantine windows remain traceable in the unified register.
- Anomaly context export: 6,534 episode-session rows, including eight explicitly
  identified missing context sessions; no missing session is converted to zero.
- Final historical-eligibility panel: 488 securities, 486 issuers, and
  1,186,196 validated stock-days after excluding 1,601 pre-policy FERG
  observations and 189 pre-policy DKNG observations.
- Final zero-volume diagnostics: 16 observations across 13 securities after
  the eligibility corrections. Extreme-volume and extreme-return counts are
  recomputed to 235 and 45 observations, respectively; extreme adjusted-price
  observations remain zero.
- Read-only 488-security identity audit: 482 securities have no detected
  conflict, six require human review, and one ticker-reuse case is verified.
  The Yahoo `B` checkpoint contains Barrick Mining rather than the intended
  Barnes Group identity; no source observation was reassigned or changed.
- Final v3 research panel: 487 securities, 485 distinct SEC issuers, and
  1,183,682 validated stock-days after removing the verified wrong-issuer `B`
  series. The v2 file and the original Yahoo `B` checkpoint remain preserved.
- OHLC-enriched v3 panel: 486 securities and 1,181,168 stock-days after a
  documented exclusion of HUBB following its OHLC consistency review. All retained rows have
  local raw Open/High/Low/Close, dividends, and stock splits, giving 100% OHLC
  coverage with no remaining OHLC issue rows.

The complete 3 January 2023 exchange-listed denominator is unavailable from the
accessible sources, so historical-universe coverage is unknown rather than
reported as a percentage. The Yahoo panel is substantial but excludes 1,635
verified securities for which Yahoo returned no usable history.

## Main outputs

- Universe audit table: `data/universe/universe_2023.csv`
- Unique acquisition security master:
  `data/universe/security_master.parquet`
- Universe issue register: `data/universe/universe_issues.csv`
- Yahoo manifest and failures: `data/metadata/download_manifest.csv`,
  `data/metadata/download_failures.csv`
- Secondary-sample manifest and failures:
  `data/metadata/eodhd_demo_manifest.csv`,
  `data/metadata/eodhd_demo_failures.csv`
- Validated research-facing base: `data/validated/base.parquet`
- Diagnostic returns only: `data/validated/diagnostic_returns.parquet`
- Per-security quality: `data/validated/data_quality_by_security.csv`
- Per-year quality: `data/validated/data_quality_by_year.csv`
- Validation issue register: `data/validated/data_issues.csv`
- Reconciled P0/P1 issue register: `data/issues/p0_p1_issue_summary.csv`
- Consolidated evidence register: `data/issues/investigation_issue_register.csv`
- Dataset readiness: `data/issues/dataset_readiness.csv`
- Investigation evidence: `data/investigation/`
- Report statistics: `reports/latex/generated/report_statistics.json`
- LaTeX source: `reports/latex/main.tex`
- Compiled report: `reports/latex/main.pdf`
- Liquidity-filter configuration: `configs/liquidity_filter.yaml`
- Filtered selected master and daily panel:
  `data/filtered/adtv100m_2023/selected_security_master.parquet`,
  `data/filtered/adtv100m_2023/selected_equity_daily.parquet`
- Liquidity screen and metadata:
  `data/filtered/adtv100m_2023/liquidity_screen_all.csv`,
  `data/filtered/adtv100m_2023/filter_metadata.json`
- Post-filter audit outputs:
  `data/filtered/adtv100m_2023/quality/`
- Filtered anomaly episodes:
  `data/investigation/filtered_anomaly_episodes/`
- Historical eligibility policy:
  `data/metadata/historical_eligibility_policy.csv`
- Final corrected research panel:
  `data/filtered/adtv100m_2023_eligibility_v2/`
- Final LaTeX report source and PDF:
  `reports/final_data_quality_report/main.tex`,
  `reports/final_data_quality_report/main.pdf`
- Identity audit outputs and cached SEC metadata:
  `data/investigation/security_identity_audit_488/`
- Final v3 dataset after verified identity exclusion:
  `data/filtered/adtv100m_2023_eligibility_v3/`
- OHLC-enriched final dataset:
  `data/filtered/adtv100m_2023_eligibility_v3_ohlc/`
- Supplementary LaTeX source and PDF:
  `reports/liquidity_report/main.tex`,
  `reports/liquidity_report/main.pdf`

Raw Yahoo files are retained under `data/raw/yahoo/` when available. Public-demo
response bodies and normalized daily files are retained separately under
`data/raw/eodhd_demo/`.

## Reproduce

Create the Anaconda environment and run the test suite:

```bash
conda env create -f environment.yml
conda activate dsa5205_proj2
python -m pytest -q
```

Build the historical universe. Set a descriptive SEC user-agent with contact
information:

```bash
export SEC_USER_AGENT='Research project name and contact email'
python scripts/build_universe.py
```

Attempt Yahoo acquisition for the full verified universe:

```bash
python scripts/download_data.py \
  --universe data/universe/security_master.parquet \
  --batch-size 50
```

The command is resumable. Do not repeatedly relaunch it after a provider-wide
rate-limit response.

Optionally acquire the clearly labelled public-demo cross-check:

```bash
python scripts/download_eodhd_sample.py
```

Validate the Yahoo data while preserving failed-download records:

```bash
python scripts/validate_data.py \
  --config configs/acquisition.yaml \
  --raw-dir data/raw/yahoo \
  --master data/universe/security_master.parquet \
  --output-dir data/validated \
  --metadata-dir data/metadata \
  --manifest data/metadata/download_manifest.csv \
  --failures data/metadata/download_failures.csv
```

Re-run the P0/P1 investigations, regenerate EDA/report tables, enforce the
source-consistency gate, and compile the report:

```bash
python scripts/run_identity_temporal.py
python scripts/run_missing_analysis.py
python scripts/investigate_volume.py
python scripts/run_anomaly_investigation.py
python scripts/build_issue_register.py
python scripts/run_eda.py
python scripts/check_report_consistency.py
python scripts/integrate_investigations.py
cd reports/latex
latexmk -pdf -interaction=nonstopmode main.tex
```

`make all` runs the full reproducible chain, including the resumable Yahoo
attempt. `make yahoo` runs only the Yahoo stage.

Build the fixed liquidity cohort, audit it, generate the report assets, enforce
the 39-check consistency gate, and compile the supplementary PDF:

```bash
make liquidity
make test
```

The equivalent direct commands are:

```bash
python scripts/build_liquidity_filter.py --config configs/liquidity_filter.yaml
python scripts/run_post_filter_anomalies.py --config configs/liquidity_filter.yaml
python scripts/run_post_filter_integrity.py --config configs/liquidity_filter.yaml
python scripts/build_liquidity_report.py --config configs/liquidity_filter.yaml
python scripts/check_liquidity_report_consistency.py
cd reports/liquidity_report
latexmk -pdf -interaction=nonstopmode main.tex
```

Build strict filtered anomaly episodes and their context-window extract:

```bash
make anomaly-episodes
make test
```

The equivalent direct command is:

```bash
python scripts/build_anomaly_episodes.py \
  --config configs/anomaly_episodes.yaml
```

Build and verify the final historical-eligibility dataset and report:

```bash
make final-quality
make test
```

The equivalent direct commands are:

```bash
python scripts/build_historical_eligibility.py \
  --config configs/historical_eligibility.yaml
python scripts/build_final_quality_report.py \
  --config configs/historical_eligibility.yaml
python scripts/check_final_quality_report_consistency.py \
  --config configs/historical_eligibility.yaml
cd reports/final_data_quality_report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Run the read-only identity audit using the cached SEC submission metadata:

```bash
make identity-audit
```

The equivalent direct command is:

```bash
python scripts/audit_security_identity_488.py \
  --config configs/security_identity_audit.yaml
```

Build the final v3 research dataset by removing the verified `B` identity
failure:

```bash
make final-dataset-v3
make test
```

The equivalent direct command is:

```bash
python scripts/build_final_dataset_v3.py \
  --config configs/final_dataset_v3.yaml
```

Enrich the final v3 dataset with locally acquired OHLC:

```bash
make ohlc-enrichment-v3
make test
```

The equivalent direct command is:

```bash
python scripts/build_ohlc_enrichment_v3.py \
  --config configs/ohlc_enrichment_v3.yaml
```

## Methodology documents

- `docs/universe_methodology.md`
- `docs/acquisition_methodology.md`
- `docs/data_quality_methodology.md`
- `docs/eodhd_sample.md`
- `docs/reporting_plan.md`
- `docs/schema.md`
- `docs/investigation_identity.md`
- `docs/investigation_missing.md`
- `docs/investigation_anomalies.md`
- `docs/investigation_volume.md`
- `docs/anomaly_episodes.md`
- `docs/p0_p1_investigation_summary.md`
- `docs/changelog.md`

## Key limitations

- SEC filings are evidence of registrant and covered-security status, not an
  official date-specific exchange census.
- Some historical ticker, listing, delisting, REIT, and security-class
  classifications remain unresolved or incomplete.
- Yahoo returned no usable history for 1,635 verified securities. These gaps
  remain in the acquisition register and may be non-random.
- Ticker-change cases are disproportionately unavailable: 379 of 391 lack a
  Yahoo history. A targeted probe found data for 12 later symbols, but those
  histories remain outside the primary base pending alias-continuity checks.
- Extreme return diagnostics require quarantine handling for near-zero
  denominators and unresolved discontinuities. Raw observations are retained.
- Subsequent research should use the explicit dataset-readiness register rather
  than treating the entire raw panel as immediately model-ready.
- The secondary public-demo source exposes only three complete ten-year
  histories and is retained only as a cross-check.
- Corporate-action checks screen continuity but cannot resolve every vendor
  convention or market event.
- Reported volume is retained in share units; no split-adjusted volume feature
  is created.
