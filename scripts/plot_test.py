"""Plot the completed fixed-model research test."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(ROOT / 'results/test/primary474_daily_0.002.csv', parse_dates=['date'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    for column, label, color in [('nav', '60/40 strategy', '#173d62')]:
        nav = d[column].to_numpy() / 1_000_000
        peak = np.maximum.accumulate(np.r_[1., nav])[1:]
        axes[0].plot(d.date, nav, label=label, color=color, lw=1.7)
        axes[1].plot(d.date, nav / peak - 1, color=color, lw=1.3)
    axes[0].set_title('2024–2025 fixed-model test | primary 474 securities', loc='left', fontsize=14)
    axes[0].set_ylabel('NAV / initial capital')
    axes[0].legend(frameon=False)
    axes[1].set_ylabel('Drawdown')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for a in axes:
        a.grid(alpha=.18)
        a.spines[['top', 'right']].set_visible(False)
    fig.text(.08, .015, '20 bps per side. Frozen 2023 model; no test refit.', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(ROOT / 'docs/figures/test_results.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
