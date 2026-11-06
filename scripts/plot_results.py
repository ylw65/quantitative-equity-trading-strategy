"""Render historical portfolio curves and cost sensitivity from local results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.6), gridspec_kw={'height_ratios': [2, 1]})
    for col, (path, title) in enumerate([
        ('training/daily_0.002.csv', '2019–2022 | walk-forward training'),
        ('validation/primary474_daily_0.002.csv', '2023 | exposed development validation')]):
        d = pd.read_csv(ROOT / 'results' / path, parse_dates=['date'])
        # Include initial transaction costs relative to the configured initial capital.
        ax[0, col].plot(d.date, d.nav / 1_000_000, color='#173d62', lw=1.6)
        ax[0, col].set_title(title, loc='left', fontsize=11)
        ax[0, col].set_ylabel('NAV / initial capital')
        dd = d.nav / d.nav.cummax() - 1
        ax[1, col].fill_between(d.date, dd, 0, color='#ac4a45', alpha=.7)
        ax[1, col].yaxis.set_major_formatter(PercentFormatter(1))
        ax[1, col].set_ylabel('Drawdown')
        for a in ax[:, col]:
            a.grid(alpha=.18)
            a.tick_params(axis='x', rotation=20)
    fig.suptitle('Quantitative Equity Trading Strategy | Volume Reversal & ML Filtering', x=.07, ha='left', fontsize=14)
    fig.text(.07, .015, '20 bps per side. Simulated adjusted-close execution. 2023 is not an independent test.', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, .04, 1, .94])
    target = ROOT / 'docs/figures'
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / 'historical_results.png', dpi=180)
    plt.close(fig)
    print('Saved docs/figures/historical_results.png')


if __name__ == '__main__':
    main()
