"""Enrich the approved v3 security-date grid with locally acquired OHLC."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.ohlc_enrichment import main


if __name__ == "__main__":
    main()
