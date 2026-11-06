"""Build strict, calendar-aware episodes from filtered anomaly evidence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.investigation.anomaly_episodes import main


if __name__ == "__main__":
    main()
