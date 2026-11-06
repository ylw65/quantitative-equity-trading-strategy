"""Run snapshot assertions locally; report explicit skips in the source-only ZIP."""
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_CASES = {
    'test_real_snapshot_date_boundary_and_policy',
    'test_data_current_identity_and_schema',
}


def pytest_collection_modifyitems(items):
    config = yaml.safe_load((ROOT / 'configs/strategy.yaml').read_text())
    # A partially supplied directory is an error, not a reason to hide a test.
    if (ROOT / config['data_directory']).exists():
        return
    for item in items:
        if item.originalname in SNAPSHOT_CASES:
            item.add_marker(pytest.mark.skip(reason='Market snapshot excluded from source-only package'))
