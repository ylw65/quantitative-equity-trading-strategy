PYTHON ?= python

.PHONY: universe yahoo secondary validate investigation consistency integrate eda report liquidity-filter post-filter-anomalies post-filter-integrity liquidity-report-assets liquidity-consistency liquidity-report liquidity anomaly-episodes historical-eligibility final-quality-assets final-quality-consistency final-quality identity-audit final-dataset-v3 ohlc-enrichment-v3 test all

universe:
	$(PYTHON) scripts/build_universe.py

yahoo:
	$(PYTHON) scripts/download_data.py --universe data/universe/security_master.parquet --batch-size 50

secondary:
	$(PYTHON) scripts/download_eodhd_sample.py

validate:
	$(PYTHON) scripts/validate_data.py --config configs/acquisition.yaml --raw-dir data/raw/yahoo --master data/universe/security_master.parquet --output-dir data/validated --metadata-dir data/metadata --manifest data/metadata/download_manifest.csv --failures data/metadata/download_failures.csv

investigation:
	$(PYTHON) scripts/run_identity_temporal.py
	$(PYTHON) scripts/run_missing_analysis.py
	$(PYTHON) scripts/investigate_volume.py
	$(PYTHON) scripts/run_anomaly_investigation.py
	$(PYTHON) scripts/build_issue_register.py

eda:
	$(PYTHON) scripts/run_eda.py

consistency:
	$(PYTHON) scripts/check_report_consistency.py

integrate:
	$(PYTHON) scripts/integrate_investigations.py

report: consistency integrate
	cd reports/latex && latexmk -pdf -interaction=nonstopmode main.tex

liquidity-filter:
	$(PYTHON) scripts/build_liquidity_filter.py --config configs/liquidity_filter.yaml

post-filter-anomalies:
	$(PYTHON) scripts/run_post_filter_anomalies.py --config configs/liquidity_filter.yaml

post-filter-integrity:
	$(PYTHON) scripts/run_post_filter_integrity.py --config configs/liquidity_filter.yaml

liquidity-report-assets:
	$(PYTHON) scripts/build_liquidity_report.py --config configs/liquidity_filter.yaml

liquidity-consistency: liquidity-report-assets
	$(PYTHON) scripts/check_liquidity_report_consistency.py

liquidity-report: liquidity-consistency
	cd reports/liquidity_report && latexmk -pdf -interaction=nonstopmode main.tex

liquidity: liquidity-filter post-filter-anomalies post-filter-integrity liquidity-report

anomaly-episodes:
	$(PYTHON) scripts/build_anomaly_episodes.py --config configs/anomaly_episodes.yaml

historical-eligibility:
	$(PYTHON) scripts/build_historical_eligibility.py --config configs/historical_eligibility.yaml

final-quality-assets: historical-eligibility
	$(PYTHON) scripts/build_final_quality_report.py --config configs/historical_eligibility.yaml

final-quality-consistency: final-quality-assets
	$(PYTHON) scripts/check_final_quality_report_consistency.py --config configs/historical_eligibility.yaml

final-quality: final-quality-consistency
	cd reports/final_data_quality_report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

identity-audit:
	$(PYTHON) scripts/audit_security_identity_488.py --config configs/security_identity_audit.yaml

final-dataset-v3:
	$(PYTHON) scripts/build_final_dataset_v3.py --config configs/final_dataset_v3.yaml

ohlc-enrichment-v3:
	$(PYTHON) scripts/build_ohlc_enrichment_v3.py --config configs/ohlc_enrichment_v3.yaml

test:
	$(PYTHON) -m pytest -q

all: universe yahoo secondary validate investigation eda consistency report test
