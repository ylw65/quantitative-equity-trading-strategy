#!/usr/bin/env python3
"""Generate report macros, tables, and figures from the filtered snapshot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.liquidity_report import main


if __name__ == "__main__":
    main()
