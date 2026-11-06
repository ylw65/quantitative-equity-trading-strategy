#!/usr/bin/env python3
"""CLI entry point for validation and data-quality auditing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.validate_data import main


if __name__ == "__main__":
    raise SystemExit(main())
