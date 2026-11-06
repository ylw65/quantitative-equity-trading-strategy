#!/usr/bin/env python3
"""Generate integrated P0/P1 report artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.investigation.integrate_report import main


if __name__ == "__main__":
    main()
