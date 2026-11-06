"""Separate portable tests from assertions against unavailable historical artifacts.

The upstream tests are retained unchanged. Artifact tests require the complete
original acquisition/audit workspace; opting in runs their original assertions.
"""
import pytest

ARTIFACT_MODULES = {'test_final_dataset_v3.py'}
ARTIFACT_CASES = {
    'test_eda_statistics_match_primary_artifacts',
    'test_generated_report_tables_and_figures_are_present',
    'test_audit_covers_exactly_selected_488_securities',
    'test_barnes_barrick_is_verified_wrong_issuer',
    'test_ferg_and_dkng_are_distinct_event_types',
    'test_events_and_ids_reconcile_without_cross_issuer_merging',
    'test_summary_and_review_register_reconcile',
    'test_required_audit_artifacts_exist',
    'test_output_files_and_reconciliations',
    'test_final_dataset_and_report_outputs', 'test_exact_deliverables_exist',
    'test_v3_security_date_grid_is_exactly_preserved',
    'test_existing_v3_fields_are_unchanged',
    'test_v3_master_and_exclusion_boundaries_are_preserved',
    'test_ohlc_completeness_and_consistency_register',
    'test_source_provenance_and_schema_comparison',
    'test_required_artifacts_exist', 'test_hubb_security_exclusion_audit',
}


def pytest_addoption(parser):
    parser.addoption('--run-artifact-tests', action='store_true', default=False,
                     help='Run assertions requiring original historical data and audit files')


def pytest_configure(config):
    config.addinivalue_line('markers', 'historical_artifact: requires original data/audit workspace')


def pytest_collection_modifyitems(config, items):
    for item in items:
        needs_artifacts = (item.path.name in ARTIFACT_MODULES or
                           item.originalname in ARTIFACT_CASES)
        if not needs_artifacts:
            continue
        item.add_marker(pytest.mark.historical_artifact)
        if not config.getoption('--run-artifact-tests'):
            item.add_marker(pytest.mark.skip(
                reason='Original historical artifacts not bundled; opt in with --run-artifact-tests'))
