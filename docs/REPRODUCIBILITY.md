# Reproduction guide

## Environments

Use Python 3.13 and `environment-lock.txt` for strategy training and inference.
The joblib snapshot requires the pinned scikit-learn environment; load only
the supplied, hash-verified model. Generate plots in the separate cleaning
environment or a separate environment using `requirements-plot.txt`.
The acquisition pipeline has separate dependencies in
`data_pipeline/requirements.txt` and `environment.yml`. Run its tests in a
separate process from the strategy tests because the two upstream projects
use different `src` import layouts.

Cleaning uses a separate environment from the frozen strategy.
Inside `data_pipeline/`:

```bash
python -m pip install -r environment-lock.txt
python -m pytest -q -p no:cacheprovider
```

`data_pipeline/environment-lock.txt` records the independently tested cleaning
environment; `requirements.txt` retains the original allowed dependency ranges.
The portable suite reports 68 passes and 29 explicit skips. Twenty-five tests
are marked `historical_artifact` because their required
historical data and audit files are not distributed. Four upstream tests also
skip when generated liquidity outputs are absent. With the full original
workspace restored, add `--run-artifact-tests` to exercise those assertions.
Skipped artifact checks are reported separately from passing tests.
The optional `python scripts/run_validation_fixture.py` exercises acquisition-
shaped synthetic data through validation without contacting a vendor. Its
2016–2025 dates are synthetic and have no held-out market information.

## Reproduction levels

1. **Public source tests:** synthetic signal, execution and model chronology
   tests do not require the private market panel (18 pass; two snapshot
   assertions skip when the data directory is absent). Cleaning tests include both
   synthetic and historical-artifact assertions; see the validation report for
   the precise executed subset and unavailable inputs.
2. **Local snapshot backtest:** the research workspace includes the final
   OHLC snapshot. Run `python scripts/verify_data.py`, then `python run.py` and
   `python run.py validation`. Training rebuilds completed-trade labels, annual
   models, predictions and the historical gate. `models/` is never overwritten.
   `python scripts/verify_reference.py` compares nine scenarios to frozen values.
3. **Raw acquisition through cleaning:** start inside `data_pipeline/` with its
   environment. Follow its README in order: universe, acquisition, validation,
   investigations, liquidity screen, eligibility and identity audit, final v3,
   OHLC enrichment. Required original raw/checkpoint files, intermediate panels,
   evidence registers and SEC caches are not bundled in this repository.
   A fresh acquisition is a new data vintage and is not guaranteed to reproduce
   the recorded panel or performance. This level was not executed here.

The final cleaning commands are:

```bash
cd data_pipeline
python scripts/build_final_dataset_v3.py --config configs/final_dataset_v3.yaml
python scripts/build_ohlc_enrichment_v3.py --config configs/ohlc_enrichment_v3.yaml
cd ..
python scripts/handoff_data.py
python scripts/verify_data.py
```

Handoff refuses to overwrite an existing strategy snapshot. If rebuilding a
different vintage, preserve its provenance and regenerate reference results
under an explicitly new research version rather than treating a hash mismatch
as a successful reproduction. The historical manifest remains immutable.

## Data boundary

The cleaning source describes 2016–2025 data. Strategy defaults to 2016–2022
and filters dates using Arrow before materialisation. Only the explicit
validation command admits 2023; neither stage reads 2024–2025 rows. The
`python run.py test` command evaluates 2024–2025 using
the unchanged frozen 2023 model. Run `python scripts/verify_reference.py all`
to check the original nine plus six test scenarios. The default reproduction
command still runs training and validation only. See TEST_REPORT.md.
File hashing reads bytes for integrity; it does not evaluate held-out prices.

## GitHub source package

`data_current/`, `data_pipeline/data/`, caches, generated annual models and
`results/` are ignored. The release ZIP contains source, frozen inference
model, methodology, test code and small verified results/figures. It excludes
raw vendor data and local Python environments.
