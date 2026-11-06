from __future__ import annotations

import pandas as pd

from src.filtering.post_filter_anomalies import (
    diagnostic_metrics,
    filter_artifact,
    selected_id_set,
)


def test_selected_ids_and_artifact_filtering() -> None:
    master = pd.DataFrame({"security_id": ["A", "B"]})
    artifact = pd.DataFrame(
        {"security_id": ["A", "B", "C"], "value": [1, 2, 3]}
    )
    ids = selected_id_set(master)
    assert ids == {"A", "B"}
    assert filter_artifact(artifact, ids)["security_id"].tolist() == ["A", "B"]


def test_diagnostic_metrics_are_cohort_rates() -> None:
    diagnostics = pd.DataFrame(
        {
            "security_id": ["A", "A", "B", "C"],
            "extreme_return_flag": [True, False, True, True],
            "extreme_volume_spike_flag": [False, True, False, True],
            "zero_volume_flag": [False, False, False, True],
        }
    )
    result = diagnostic_metrics(diagnostics, {"A", "B"})
    assert result["rows"] == 3
    assert result["extreme_returns"] == 2
    assert result["extreme_return_rate"] == 2 / 3
    assert result["zero_volume_rows"] == 0
    assert result["extreme_volume_spikes"] == 1
