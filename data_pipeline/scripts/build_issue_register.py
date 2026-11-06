#!/usr/bin/env python3
"""Consolidate investigation issue registers without discarding evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import PROJECT_ROOT


def main() -> None:
    sources = [
        ("identity", PROJECT_ROOT / "data/investigation/identity/identity_issues.csv"),
        (
            "missing",
            PROJECT_ROOT / "data/investigation/missing/missing_security_issues.csv",
        ),
        ("anomaly", PROJECT_ROOT / "data/investigation/anomalies/anomaly_issues.csv"),
        ("volume", PROJECT_ROOT / "data/investigation/volume/volume_issues.csv"),
    ]
    frames = []
    for scope, path in sources:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        frame.insert(0, "investigation_scope", scope)
        frames.append(frame)
    register = pd.concat(frames, ignore_index=True, sort=False)
    summary = pd.read_csv(
        PROJECT_ROOT / "data/issues/p0_p1_issue_summary.csv",
        low_memory=False,
    )
    summary.insert(0, "investigation_scope", "p0_p1_summary")
    register = pd.concat([summary, register], ignore_index=True, sort=False)
    output = PROJECT_ROOT / "data/issues/investigation_issue_register.csv"
    register.to_csv(output, index=False)
    print(output)
    print(f"rows={len(register)}")


if __name__ == "__main__":
    main()
