"""Generate the final historical-eligibility and data-quality report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.final_quality_report import main


if __name__ == "__main__":
    main()
