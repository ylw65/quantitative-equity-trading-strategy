from __future__ import annotations

import pandas as pd

from src.investigation.volume_analysis import (
    longest_true_run,
    zero_concentration,
)


def test_longest_true_run() -> None:
    values = pd.Series([False, True, True, False, True, True, True, False])
    assert longest_true_run(values) == 3
    assert longest_true_run(pd.Series([], dtype=bool)) == 0
    assert longest_true_run(pd.Series([False, False])) == 0


def test_zero_concentration_counts_securities_not_rows() -> None:
    stats = zero_concentration(pd.Series({"A": 50, "B": 30, "C": 20, "D": 0}))
    assert stats["securities_with_zero_volume"] == 3
    assert stats["zero_volume_rows"] == 100
    assert stats["securities_for_first_half"] == 1
    assert stats["max_zero_rows_one_security"] == 50
