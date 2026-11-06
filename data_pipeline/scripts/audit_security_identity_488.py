"""Run the read-only security identity audit for the selected 488 securities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.investigation.security_identity_audit import main


if __name__ == "__main__":
    main()
