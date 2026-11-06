#!/usr/bin/env python3
"""Run the P1 trading-volume integrity investigation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.investigation.volume_analysis import main


if __name__ == "__main__":
    main()
