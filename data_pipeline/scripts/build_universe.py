#!/usr/bin/env python3
"""Run the historical universe build from the project root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.universe.build_universe import main


if __name__ == "__main__":
    main()
