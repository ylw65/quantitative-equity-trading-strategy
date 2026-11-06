import copy
import pandas as pd
import pytest
from market import check_window
from strategy import config
from pipeline import build_teacher


def test_explicit_test_mode_is_bounded_and_default_stays_training_only():
    c = config()
    assert check_window(c)[1] == pd.Timestamp('2022-12-31')
    future = copy.deepcopy(c)
    future['allowed_end'] = '2025-12-31'
    with pytest.raises(ValueError, match='explicit test mode'):
        check_window(future)
    future['test_enabled'] = True
    assert check_window(future)[1] == pd.Timestamp('2025-12-31')
    future['allowed_end'] = '2026-01-01'
    with pytest.raises(ValueError, match='authorised test period'):
        check_window(future)


def test_test_data_cannot_build_training_labels():
    with pytest.raises(ValueError, match='Never build learning labels'):
        build_teacher({'dates': pd.DatetimeIndex(['2024-01-02'])}, None,
                      config(), None, None)
