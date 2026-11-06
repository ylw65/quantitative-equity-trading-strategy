"""Build the final research panel after removing verified identity failure B."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.filtering.final_dataset_v3 import main


if __name__ == "__main__":
    main()
