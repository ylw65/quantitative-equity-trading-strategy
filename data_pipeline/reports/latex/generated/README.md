# Generated report inputs

The EDA/report integration step writes empirical text fragments and tables here.
No file in this directory may contain invented values.

## Required prose fragments

- `executive_summary.tex`
- `universe_method.tex`
- `universe_results.tex`
- `acquisition_results.tex`
- `coverage_results.tex`
- `missingness_results.tex`
- `corporate_action_results.tex`
- `volume_results.tex`
- `extreme_results.tex`
- `identity_bias_results.tex`
- `cleaning_decisions.tex`
- `limitations.tex`
- `conclusions.tex`
- `configuration.tex`
- `reproduction_commands.tex`

Each fragment must be valid LaTeX body text and must distinguish verified
findings from possible explanations. It should cite the exact source table or
figure supporting material claims.

## Generated table fragments

Each table fragment is a complete `tabular` environment generated from the
corresponding artifact:

- `table_universe_by_exchange.tex`
- `table_universe_by_type.tex`
- `table_download_status.tex`
- `table_coverage_by_year.tex`
- `table_missingness_by_year.tex`
- `table_gap_examples.tex`
- `table_action_counts.tex`
- `table_dividend_audit.tex`
- `table_split_audit.tex`
- `table_volume_flags.tex`
- `table_extreme_returns.tex`
- `table_identity_issues.tex`
- `table_issue_summary.tex`

## Required figures

Figures are vector PDFs under `reports/figures/`:

- `universe_composition.pdf`
- `issue_composition.pdf`
- `download_status.pdf`
- `observations_by_year.pdf`
- `available_securities_by_year.pdf`
- `representative_histories.pdf`
- `missingness_distribution.pdf`
- `corporate_actions_by_year.pdf`
- `split_adjustment_audit.pdf`
- `volume_distribution_log.pdf`
- `adjusted_close_distribution.pdf`
- `diagnostic_returns_tail.pdf`

The integration run is expected to replace all pending notices with generated
artifacts. `report_statistics.json` is the machine-readable numerical source;
`macros.tex` supplies counts used in report prose.
