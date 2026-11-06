"""Copy the final cleaning output into the strategy input directory."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'adtv100m_2023_eligibility_v3_ohlc'
REQUIRED = ['selected_equity_daily.parquet', 'selected_security_master.parquet',
            'dataset_metadata.json', 'ohlc_source_provenance.json']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path,
                   default=ROOT / 'data_pipeline/data/filtered' / NAME)
    args = p.parse_args()
    source = args.source.resolve()
    target = ROOT / 'data_current' / NAME
    if not all((source / name).is_file() for name in REQUIRED):
        raise SystemExit(f'Incomplete OHLC dataset: {source}')
    if target.exists():
        raise SystemExit('Destination already exists; preserve it before importing a new snapshot.')
    target.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, target / file.name)
    print(f'Copied final cleaning output to {target}.')


if __name__ == '__main__':
    main()
