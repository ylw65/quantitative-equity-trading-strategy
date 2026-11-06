#!/usr/bin/env python3
"""Run the P1 extreme-price and corporate-action investigation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.investigation.anomaly_analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
