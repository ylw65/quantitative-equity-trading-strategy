"""Verify the supplied snapshot and rebuild training and validation periods."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for args in [
        ['scripts/verify_data.py'],
        ['-m', 'pytest', '-q'],
        ['run.py'],
        ['run.py', 'validation'],
        ['scripts/verify_reference.py'],
    ]:
        print('Running:', ' '.join(args), flush=True)
        subprocess.run([sys.executable, '-B', *args], cwd=ROOT, check=True)
    print('Complete: training and exposed 2023 validation reproduced; test excluded.')


if __name__ == '__main__':
    main()
