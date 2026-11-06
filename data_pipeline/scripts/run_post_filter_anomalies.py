#!/usr/bin/env python3
"""Run post-filter anomaly, volume, and corporate-action assessment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.post_filter_anomalies import main


if __name__ == "__main__":
    main()
