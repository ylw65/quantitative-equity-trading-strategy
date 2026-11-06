#!/usr/bin/env python3
"""Validate the liquidity report against its filtered-data snapshot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.liquidity_report_consistency import main


if __name__ == "__main__":
    raise SystemExit(main())
