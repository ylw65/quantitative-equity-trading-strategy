"""Compare generated results with the final selected research record; never fits."""
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=['training', 'validation', 'test', 'both', 'all'], default='both')
    stage = parser.parse_args().stage
    reference = pd.read_csv(ROOT / 'docs/final_results.csv')
    if stage in ['test', 'all']:
        test_reference = pd.read_csv(ROOT / 'docs/test_results.csv')
        test_reference['stage'] = 'test'
        reference = pd.concat([reference, test_reference], ignore_index=True)
    checked = 0
    for section in ['training', 'validation', 'test']:
        if not (stage == section or stage == 'all' or (stage == 'both' and section != 'test')):
            continue
        actual = pd.read_csv(ROOT / f'results/{section}/comparison.csv')
        if section == 'training':
            actual['cohort'] = 'delivered486'
        for row in reference[reference.stage.eq(section)].itertuples():
            matched = actual[actual.cohort.eq(row.cohort) & actual.fee.eq(row.fee)]
            if len(matched) != 1:
                raise ValueError(f'Missing result: {section}/{row.cohort}/{row.fee}')
            keys = ['annual_return', 'total_return', 'volatility', 'sharpe',
                    'max_drawdown', 'average_stock', 'cost']
            for key in keys:
                tolerance = .01 if key == 'cost' else 1e-10
                if abs(float(matched.iloc[0][key]) - getattr(row, key)) > tolerance:
                    raise ValueError(f'Result mismatch: {section}/{row.cohort}/{row.fee}/{key}')
            checked += 1
    print(f'PASS: {checked} frozen reference scenarios reproduced; no market rows read.')


if __name__ == '__main__':
    main()
