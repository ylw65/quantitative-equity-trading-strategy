#!/usr/bin/env python3
"""Run the fixed ADTV20 liquidity filter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.liquidity_screen import main


if __name__ == "__main__":
    main()
