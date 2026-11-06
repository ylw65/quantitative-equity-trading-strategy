#!/usr/bin/env python3
"""Validate report statistics against acquisition and validated data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.investigation.report_consistency import main


if __name__ == "__main__":
    raise SystemExit(main())
