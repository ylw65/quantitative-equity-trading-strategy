"""Validate the final report against the v2 research snapshot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.final_quality_report_consistency import main


if __name__ == "__main__":
    raise SystemExit(main())
