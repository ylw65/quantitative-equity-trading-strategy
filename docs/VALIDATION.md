# Validation record

Cleaning and strategy source versions are pinned in `source_manifest.json`.
Validation covers data integrity, chronological controls, execution accounting
and reproducibility.

## Confirmed results

- Final daily panel and security master match the confirmed GitHub `main`
  snapshot and original strategy input by SHA-256; all ten snapshot files pass.
- Strategy suite: 20 tests covering signal behaviour, execution, features,
  labels, calibration and time boundaries.
  Source-only checkouts run 18 tests and explicitly skip the two snapshot
  assertions; partial or invalid supplied data still fails the assertions.
- Cleaning portable suite: 68 tests passed, 29 skipped. Of the skips, 25 require
  original historical artifacts and four are upstream liquidity-output checks.
  No missing-artifact test is counted as passing. Original assertions remain
  available through `--run-artifact-tests` with restored inputs.
- A fresh cleaning environment passed its dependency check. The synthetic
  validation fixture processed ten securities into 24,048 validated rows,
  with seven deliberately induced data issues recorded. No vendor request was
  made; these values are not market performance evidence.
- Training rebuilt 12,990 completed trade labels, annual classifiers and past
  calibration, reproducing frozen 2023 model predictions and threshold.
- Nine training/validation fee and universe scenarios match the reference
  record: metric tolerance 1e-10, dollar-cost tolerance USD 0.01.
- Default model training ends in 2022; 2023 uses frozen inference. Strategy
  training/validation reject test dates before materialisation; explicit test mode
  admits 2024–2025 and rejects dates after 2025. Two additional tests check
  evaluation date boundaries and the prohibition on test-derived training labels.
- The fixed-model test evaluated 2024–2025 without refitting and
  recorded six predefined fee/universe scenarios and annual returns. All fifteen historical/reference scenarios match. See TEST_REPORT.md.

## Scope

The supplied final-data snapshot was backtested. A from-raw historical rebuild
was **not** performed: original checkpoints, SEC caches and intermediate audit
outputs are not included upstream. The acquisition/validation synthetic fixture
is separate from market performance. A fresh acquisition would be a new vintage.

CI configuration defines separate strategy and cleaning jobs.
Extracted source-package checks pass: 18 strategy tests with two explicit
snapshot skips, and 68 cleaning tests with 29 documented artifact skips.
Archive integrity, private-data exclusions and top-level documentation links
are verified. See `validation_summary.json` for a machine-readable record.

## Known dependency warnings

The frozen strategy environment reports a scikit-learn/SciPy solver-option
warning and a pandas future downcasting warning. Both were present in the
frozen implementation; they do not change the reproduced reference values.
Model hyperparameters remain fixed across the evaluation stages.
