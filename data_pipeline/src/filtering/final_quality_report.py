"""Generate final report assets from the eligibility-corrected dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, atomic_write_json, load_config, project_path


COLORS = {
    "navy": "#19324D",
    "blue": "#3F6F8F",
    "teal": "#4E8C86",
    "gold": "#B28A48",
    "rust": "#A45D4F",
    "slate": "#65727A",
    "light": "#DCE3E6",
}


def _tex(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "_": r"\_\allowbreak{}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _macro(name: str, value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        rendered = "Yes" if bool(value) else "No"
    elif isinstance(value, (int, np.integer)):
        rendered = f"{int(value):,}"
    elif isinstance(value, (float, np.floating)):
        rendered = "Unknown" if not np.isfinite(value) else f"{float(value):,.4f}"
    else:
        rendered = _tex(value)
    return rf"\newcommand{{\{name}}}{{{rendered}}}"


def _table(
    frame: pd.DataFrame,
    output_dir: Path,
    name: str,
    align: str | None = None,
) -> None:
    align = align or ("l" + "r" * max(0, len(frame.columns) - 1))
    lines = [
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(_tex(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_tex(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figures(
    *,
    final_daily: pd.DataFrame,
    exclusions: pd.DataFrame,
    zero_by_security: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )

    # Historical eligibility timeline.
    boundaries = {
        "FERG": (pd.Timestamp("2016-01-04"), pd.Timestamp("2022-05-12")),
        "DKNG": (pd.Timestamp("2019-07-25"), pd.Timestamp("2020-04-24")),
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for position, ticker in enumerate(["FERG", "DKNG"]):
        start, eligible = boundaries[ticker]
        end = final_daily.loc[
            final_daily["ticker"].eq(ticker), "date"
        ].max()
        ax.barh(
            position,
            (eligible - start).days,
            left=start,
            color=COLORS["light"],
            edgecolor=COLORS["slate"],
            height=0.42,
            label="Excluded predecessor period" if position == 0 else None,
        )
        ax.barh(
            position,
            (end - eligible).days,
            left=eligible,
            color=COLORS["teal"],
            height=0.42,
            label="Retained research history" if position == 0 else None,
        )
        ax.axvline(eligible, color=COLORS["rust"], linewidth=1.0, linestyle="--")
        ax.text(
            eligible,
            position + 0.31,
            eligible.strftime("%Y-%m-%d"),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["rust"],
        )
    ax.set_yticks([0, 1], ["FERG", "DKNG"])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlabel("Date")
    ax.set_title("Historical eligibility boundaries")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, -0.34),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_dir / "eligibility_timeline.pdf")
    plt.close(fig)

    # Remaining zero-volume concentration.
    plotted = zero_by_security.sort_values("zero_volume_rows", ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(plotted["ticker"], plotted["zero_volume_rows"], color=COLORS["gold"])
    ax.set_xlabel("Zero-volume observations")
    ax.set_title("Remaining zero-volume observations by security")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / "remaining_zero_volume_by_security.pdf")
    plt.close(fig)

    # Scale comparison. Rates keep the large acquired cohort comparable.
    metrics = summary[
        summary["metric"].isin(
            [
                "Zero-volume observations",
                "Extreme-volume observations",
                "Extreme diagnostic-return observations",
            ]
        )
    ].copy()
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for offset, column, label, color in [
        (-width, "original_acquired_per_100k", "Acquired", COLORS["slate"]),
        (0.0, "original_liquidity_filtered_per_100k", "ADTV filtered", COLORS["gold"]),
        (width, "final_research_dataset_per_100k", "Final v2", COLORS["teal"]),
    ]:
        ax.bar(x + offset, metrics[column], width, label=label, color=color)
    ax.set_xticks(
        x,
        [value.replace(" observations", "") for value in metrics["metric"]],
        rotation=12,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Observations per 100,000")
    ax.set_title("Quality-event rates across dataset scopes")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "quality_rate_comparison.pdf")
    plt.close(fig)


def _write_report_source(
    *,
    report_root: Path,
    snapshot: dict[str, Any],
    hashes: dict[str, str],
) -> None:
    sections = report_root / "sections"
    generated = report_root / "generated"
    sections.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    source_note = _tex(
        "data/filtered/adtv100m_2023_eligibility_v2 and "
        "reports/final_data_quality_report/generated/report_statistics.json"
    )
    main = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage[a4paper,margin=2.35cm,headheight=22pt]{{geometry}}
\usepackage{{microtype}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{threeparttable}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{caption}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{natbib}}
\usepackage[hidelinks]{{hyperref}}

\definecolor{{Navy}}{{HTML}}{{19324D}}
\definecolor{{Slate}}{{HTML}}{{566573}}
\definecolor{{SoftBG}}{{HTML}}{{F3F6F7}}
\definecolor{{SoftRule}}{{HTML}}{{8FA3B0}}
\graphicspath{{{{../figures/final_quality/}}}}
\captionsetup{{font=small,labelfont=bf,labelsep=period}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.55em}}
\renewcommand{{\arraystretch}}{{1.14}}
\bibliographystyle{{plainnat}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\color{{Slate}}US Equity Research Dataset}}
\fancyhead[R]{{\small\color{{Slate}}Final eligibility v2}}
\fancyfoot[C]{{\small\thepage}}
\renewcommand{{\headrulewidth}}{{0.3pt}}
\renewcommand{{\headrule}}{{\hbox to\headwidth{{\color{{SoftRule}}\leaders\hrule height \headrulewidth\hfill}}}}
\titleformat{{\section}}{{\Large\bfseries\color{{Navy}}}}{{\thesection}}{{0.8em}}{{}}
\titleformat{{\subsection}}{{\large\bfseries\color{{Navy}}}}{{\thesubsection}}{{0.7em}}{{}}
\newcommand{{\GeneratedInput}}[1]{{\input{{generated/#1.tex}}}}
\newcommand{{\GeneratedFigure}}[3]{{%
  \begin{{figure}}[tbp]
    \centering
    \includegraphics[width=\linewidth,height=0.68\textheight,keepaspectratio]{{#1}}
    \caption{{#2}}
    \label{{#3}}
  \end{{figure}}}}
\newcommand{{\GeneratedFigureH}}[3]{{%
  \begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth,height=0.68\textheight,keepaspectratio]{{#1}}
    \caption{{#2}}
    \label{{#3}}
  \end{{figure}}}}
\newcommand{{\GeneratedTable}}[1]{{\adjustbox{{max width=\linewidth}}{{\input{{generated/#1.tex}}}}}}
\GeneratedInput{{macros}}

\title{{\color{{Navy}}\textbf{{Liquidity-Based Universe Selection and\\
Final Data Quality Assessment}}}}
\author{{DSA5205 Project Team}}
\date{{Final research snapshot generated 28 October 2026}}

\begin{{document}}
\maketitle
\begin{{abstract}}
This report updates the fixed USD 100 million ADTV20 cohort with two historical
eligibility boundaries. FERG begins on 12 May 2022 under the primary-NYSE
listing policy, and DKNG begins on 24 April 2020 after the predecessor SPAC
period. The \FinalSecurities{{}} selected identities and \FinalIssuers{{}} SEC
issuers remain unchanged. The final validated panel contains
\FinalObservations{{}} stock-days after excluding \FergExcluded{{}} FERG and
\DkngExcluded{{}} DKNG observations. Remaining zero-volume and extreme-value
observations are retained with explicit unresolved-cause and evidence flags.
\end{{abstract}}
\begin{{center}}
\fcolorbox{{SoftRule}}{{SoftBG}}{{%
\begin{{minipage}}{{0.94\linewidth}}\raggedright
\small\textbf{{Final snapshot contract.}} All empirical quantities in this
report come from \texttt{{{source_note}}}. Earlier source archives are
hash-protected and remain unchanged.
\end{{minipage}}}}
\end{{center}}
\tableofcontents
\listoffigures
\listoftables
\clearpage
\input{{sections/01_summary}}
\input{{sections/02_scope}}
\input{{sections/03_universe}}
\input{{sections/04_eligibility}}
\input{{sections/05_final_dataset}}
\input{{sections/06_zero_volume}}
\input{{sections/07_extremes}}
\input{{sections/08_comparison}}
\input{{sections/09_treatments}}
\input{{sections/10_limitations}}
\input{{sections/11_conclusions}}
\appendix
\input{{sections/A_data_dictionary}}
\input{{sections/A_reproduction}}
\bibliography{{../latex/references}}
\end{{document}}
"""
    (report_root / "main.tex").write_text(main, encoding="utf-8")

    section_texts = {
        "01_summary.tex": r"""\section{Executive Summary}

The fixed liquidity cohort remains \FinalSecurities{} securities and
\FinalIssuers{} distinct SEC issuers. Historical eligibility corrections shorten
two histories but remove no security identity. The corrected research panel has
\FinalObservations{} validated stock-day observations, compared with
\FilteredObservations{} before the corrections. Exactly \FergExcluded{} FERG
observations through 2022-05-11 and \DkngExcluded{} DKNG observations through
2020-04-23 were excluded.

The corrections materially reduce the zero-volume burden: the cohort falls from
\OriginalZeroVolumeRows{} zero-volume observations to \FinalZeroVolumeRows{}
across \FinalZeroVolumeSecurities{} securities. All remaining zero values are
vendor-confirmed values but have unverified market-level causes. They are
retained and labeled \path{vendor_confirmed_market_unverified}.

The final panel contains \FinalExtremeVolumeRows{} extreme-volume observations,
no extreme adjusted-price observations under the existing evidence, and
\FinalExtremeReturnRows{} diagnostic returns beyond the absolute 50 percent
threshold. None of the retained extreme returns is classified as an unresolved
discontinuity. Statistical extremity is not treated as sufficient grounds for
deletion.
""",
        "02_scope.tex": r"""\section{Scope and Research Contract}

This phase performs data acquisition remediation, eligibility treatment, data
quality assessment, and documentation. It does not calculate final returns,
create strategy features, split the sample for model development, train models,
or run backtests. The research-facing \texttt{pct\_return} field remains null.

FERG and DKNG are excluded only from the new research-facing history before
their stated boundaries. Their original Yahoo checkpoints, validated
observations, diagnostic returns, and liquidity-screen records remain intact.
The original 488-security cohort therefore remains identifiable and is not
replaced by a current ticker directory.
""",
        "03_universe.tex": r"""\section{Universe and Liquidity Cohort}

The source universe contains \UniverseSecurities{} SEC-derived common-equity
records. Yahoo Finance provided histories for \AcquiredSecurities{} securities,
and \SelectedSecurities{} passed the fixed inclusive ADTV20 threshold using the
final 20 XNYS sessions of 2022. The v2 historical corrections do not alter the
screen, its price-volume restoration rule, or the selected identities.

The source master lacks complete verified listing and delisting dates for most
securities. The two policy rows in this report are therefore explicit
security-specific research decisions, joined by permanent security ID rather
than ticker. They should not be generalized into assumptions about every other
security.
""",
        "04_eligibility.tex": r"""\section{Historical Security Eligibility}
\label{sec:eligibility}

\subsection{Ferguson}

Ferguson's NYSE trading began on 8 March 2021. The research policy instead uses
12 May 2022 as the eligibility boundary because that is the documented
primary-listing transition \citep{secFERG2021,secFERG2022}. This is a conservative institutional-status rule,
not the first date on which FERG traded in the United States. The 1,601
observations from 2016-01-04 through 2022-05-11 are excluded from the v2
research panel; 913 observations from 2022-05-12 onward are retained.

\subsection{DraftKings}

DraftKings completed its business combination with Diamond Eagle Acquisition
Corp. on 23 April 2020, and the operating-company shares began trading as DKNG
on 24 April 2020 \citep{secDKNG2020}. The earlier Yahoo history beginning 25 July 2019 represents
the predecessor DEAC period. The 189 observations through 2020-04-23 are
excluded for economic continuity; 1,430 observations from 2020-04-24 onward are
retained. This treatment does not claim that the predecessor observations are
vendor errors.

\begin{table}[H]
\centering
\caption{Historical eligibility policy}
\label{tab:eligibility-policy}
\scriptsize
\GeneratedTable{table_eligibility_policy}
\begin{tablenotes}\small
\item Dates are inclusive. Source references are recorded in
\path{historical_eligibility_policy.csv} and the accompanying evidence file.
\end{tablenotes}
\end{table}

\begin{table}[H]
\centering
\caption{Historical exclusion reconciliation}
\label{tab:exclusions}
\GeneratedTable{table_historical_exclusions}
\end{table}

\GeneratedFigure{eligibility_timeline.pdf}{Original predecessor periods and
retained research histories for FERG and DKNG. Dashed lines mark the inclusive
eligibility boundaries.}{fig:eligibility-timeline}
""",
        "05_final_dataset.tex": r"""\section{Final Research Dataset}

The final panel contains \FinalSecurities{} securities,
\FinalIssuers{} distinct SEC issuers, and \FinalObservations{} validated
stock-day observations. Security-date keys are unique, the final
\texttt{pct\_return} field is null throughout, and all retained prices and
volumes are unchanged from the source records.

Exclusions carry the original raw source-file path and policy reference in
\path{historical_exclusions.csv}. The final master identifies whether each
security uses the default 2016--2025 history or a historical eligibility
boundary. No observations are imputed, interpolated, forward-filled, clipped,
or winsorized.

\begin{table}[tbp]
\centering
\caption{Final observations by calendar year}
\label{tab:final-years}
\small
\GeneratedTable{table_final_years}
\end{table}

The shortened FERG and DKNG histories now begin at their policy dates rather
than carrying a missing or inferred observation immediately before those dates.
Their first retained rows have null diagnostic returns because a return across
the removed boundary is intentionally not calculated.
""",
        "06_zero_volume.tex": r"""\section{Zero-Volume Investigation}

Yahoo reports zero for \OriginalZeroVolumeRows{} observations in the original
ADTV-filtered cohort. The FERG boundary removes 838 such observations; the
DKNG boundary removes 46. The final panel retains \FinalZeroVolumeRows{}
zero-volume observations across \FinalZeroVolumeSecurities{} securities,
grouped into \FinalZeroVolumeEpisodes{} strict XNYS-session episodes. The
longest remaining episode lasts \FinalZeroVolumeLongest{} trading sessions.

These are source-fidelity observations: the vendor value is zero and is not
missing. The market-level causes are not independently verified, so the
classification remains \path{vendor_confirmed_market_unverified}. Zero has
not been converted to null and a missing session has not been converted to
zero.

\GeneratedFigureH{remaining_zero_volume_by_security.pdf}{Remaining
zero-volume observations after the FERG and DKNG history corrections.}{fig:remaining-zero}

\begin{table}[H]
\centering
\caption{Remaining zero-volume observations by security}
\label{tab:remaining-zero}
\scriptsize
\GeneratedTable{table_remaining_zero}
\end{table}
""",
        "07_extremes.tex": r"""\section{Extreme-Value Investigation}

\subsection{Trading volume}

The existing definition compares reported share volume with 20 times the
median of the preceding 20 observations after at least five history
observations. Recalculation within the retained history segments yields
\FinalExtremeVolumeRows{} extreme-volume observations across
\FinalExtremeVolumeSecurities{} securities and \FinalExtremeVolumeEpisodes{}
strict episodes. Maximum reported share volume is
\FinalExtremeVolumeMaxShares{} shares. Maximum reconstructed daily dollar
volume is USD \FinalExtremeVolumeMaxDollar{}.

The total is not obtained by subtracting FERG and DKNG flags from the prior
count: truncating their histories changes the available rolling-history
baseline, so diagnostics are recomputed within each retained segment.
Extreme-volume observations remain in the dataset.

\subsection{Adjusted prices and returns}

No retained security has a nonpositive adjusted Close or an existing
extreme-price-level case record, so the final extreme-price observation count is
zero. The diagnostic return is
\(r_t=\mathrm{AdjClose}_t/\mathrm{AdjClose}_{t-1}-1\), flagged only when
\(|r_t|>0.50\). The final panel contains \FinalExtremeReturnRows{}
extreme-return observations across \FinalExtremeReturnSecurities{} securities:
\FinalExtremeReturnPositive{} positive and \FinalExtremeReturnNegative{}
negative. They form \FinalExtremeReturnEpisodes{} strict episodes.

\begin{table}[H]
\centering
\caption{Recomputed extreme-value summary}
\label{tab:extreme-summary}
\GeneratedTable{table_extreme_summary}
\end{table}

\GeneratedFigure{quality_rate_comparison.pdf}{Zero-volume, extreme-volume, and
extreme-return event rates per 100,000 observations across the acquired,
original liquidity-filtered, and final v2 scopes.}{fig:quality-rates}

\subsection{Evidence interpretation}

Existing evidence classifies \RetainedExtremeMarketMoves{} retained extreme
returns as market moves and \SameDayDividendAdjustmentCases{} as a same-day
dividend adjustment. \CorporateActionExtremeReturns{} extreme return is
corporate-action related. \OverlappingVolumeReturnEpisodes{} strict episodes
contain both extreme-volume and
extreme-return flags. No retained extreme-return discontinuity is classified as
unresolved.
""",
        "08_comparison.tex": r"""\section{Before-and-After Data Quality Comparison}

Table~\ref{tab:comparison} independently recomputes the three dataset scopes.
Rates use observations per 100,000 for event metrics. Missing adjusted-close and
missing-volume counts refer to the validated research panels, which contain
valid prices and volumes by construction; raw all-NaN placeholder rows remain a
separate vendor-archive issue.

\begin{table}[H]
\centering
\caption{Original acquired, original liquidity-filtered, and final research panels}
\label{tab:comparison}
\scriptsize
\GeneratedTable{table_quality_comparison}
\end{table}
""",
        "09_treatments.tex": r"""\section{Final Data-Treatment Decisions}

The policy distinguishes historical eligibility, economic continuity, vendor
fidelity, numerical validity, and execution eligibility. Removing a predecessor
period does not prove that its observations were numerically invalid, and
retaining a Yahoo observation does not imply that an executable trade occurred
at the recorded price.

\begin{table}[H]
\centering
\caption{Final cleaning and retention decisions}
\label{tab:treatments}
\small
\GeneratedTable{table_treatments}
\end{table}

Unresolved non-volume problems remain visible. The selected cohort retains one
ordinary-dividend adjustment mismatch, one OHLC consistency issue, two identity
review candidates, and all 16 remaining zero-volume observations with
unverified market causes.
""",
        "10_limitations.tex": r"""\section{Methodological Limitations}

The liquidity cohort is fixed using December 2022 activity. Applying that cohort
back to 2016 creates retrospective selection and can over-represent companies
that later became highly liquid. It is not a rolling point-in-time universe.

Yahoo Finance remains the primary historical source and may omit, rename, or
mis-map securities. The secondary public source does not provide broad enough
coverage to verify every observation. Historical listing dates, delisting
dates, corporate-action types, reported volume adjustments, and the causes of
remaining zero volumes are not completely resolved.

The FERG and DKNG boundaries are research decisions supported by filing
chronology and observed market-history transitions. They preserve economic
continuity but do not certify every earlier or later security identity.

Statistical outliers are not automatically data errors. The retained extreme
observations may represent economic events, corporate actions, structural
market conditions, or vendor behavior. No retained price is assumed to be
executable merely because it appears in the vendor history.
""",
        "11_conclusions.tex": r"""\section{Conclusions and Recommendations}

The corrected panel contains \FinalSecurities{} securities,
\FinalIssuers{} issuers, and \FinalObservations{} validated observations. It
preserves the fixed liquidity membership while enforcing the two documented
historical eligibility boundaries. Original vendor and validation archives
remain unchanged.

Zero-volume exposure is sharply lower after the corrections, but the remaining
16 observations still require market-level investigation. Extreme observations
remain available with explicit diagnostics. Subsequent research should treat
the final v2 panel as a documented retrospective research cohort, not as a
complete point-in-time executable universe.
""",
        "A_data_dictionary.tex": r"""\section{Data Dictionary}

\begin{longtable}{@{}p{0.40\linewidth}p{0.51\linewidth}@{}}
\toprule
Artifact or field & Definition \\
\midrule
\endhead
\path{historical_eligibility_policy.csv} & Inclusive, security-ID-specific eligibility boundaries and provenance. \\
\path{selected_equity_daily.parquet} & Final research-facing daily fields after historical exclusions. \\
\path{selected_security_master.parquet} & Fixed 488-security master with the applied historical policy metadata. \\
\path{historical_exclusions.csv} & Every excluded observation and its original raw source file. \\
\path{diagnostic_returns.parquet} & Recomputed audit-only returns and existing diagnostic flags. \\
\path{remaining_zero_volume_cases.csv} & Retained zero-volume observations and strict episode IDs. \\
\path{remaining_extreme_value_cases.csv} & Retained volume, price, and return extremes with evidence fields. \\
\path{final_data_quality_summary.csv} & Independently recomputed comparison metrics for all three scopes. \\
\path{cleaning_policy.json} & Final treatments, source evidence, policy, software versions, and input hashes. \\
\bottomrule
\end{longtable}
""",
        "A_reproduction.tex": r"""\section{Reproduction Instructions}

Run from the project root after activating the Python 3.11 environment:

\begin{verbatim}
conda activate dsa5205_proj2
python scripts/build_historical_eligibility.py \
  --config configs/historical_eligibility.yaml
python scripts/build_final_quality_report.py \
  --config configs/historical_eligibility.yaml
python scripts/check_final_quality_report_consistency.py \
  --config configs/historical_eligibility.yaml
cd reports/final_data_quality_report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cd ../..
python -m pytest -q
\end{verbatim}

The dataset build and report assets are also available through
\texttt{make final-quality}, followed by \texttt{make test}.
""",
    }
    for name, text in section_texts.items():
        (sections / name).write_text(text, encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(project_path(config_path))
    filtered_root = project_path(config["outputs"]["root"])
    report_root = project_path(config["outputs"]["report_source"])
    table_dir = project_path(config["outputs"]["tables"])
    figure_dir = project_path(config["outputs"]["figures"])
    generated = report_root / "generated"
    report_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)

    final_daily = pd.read_parquet(filtered_root / "selected_equity_daily.parquet")
    master = pd.read_parquet(filtered_root / "selected_security_master.parquet")
    exclusions = pd.read_csv(
        filtered_root / "historical_exclusions.csv", low_memory=False
    )
    zero_cases = pd.read_csv(
        filtered_root / "remaining_zero_volume_cases.csv", low_memory=False
    )
    zero_by_security = pd.read_csv(
        filtered_root / "remaining_zero_volume_by_security.csv", low_memory=False
    )
    extremes = pd.read_csv(
        filtered_root / "remaining_extreme_value_cases.csv", low_memory=False
    )
    quality = pd.read_csv(
        filtered_root / "final_data_quality_summary.csv", low_memory=False
    )
    cleaning = json.loads(
        (filtered_root / "cleaning_policy.json").read_text(encoding="utf-8")
    )
    snapshot = cleaning["snapshot"]
    policy = pd.DataFrame(cleaning["policy"])
    original = pd.read_parquet(
        project_path(config["inputs"]["original_selected_daily"])
    )
    original_daily = original.merge(
        master[["security_id", "ticker_2023"]],
        on="security_id",
        how="left",
    )
    original_zero = {
        ticker: int(
            original_daily.loc[
                original_daily["ticker_2023"].eq(ticker), "volume"
            ].eq(0).sum()
        )
        for ticker in ("FERG", "DKNG")
    }

    final_zero = int(final_daily["volume"].eq(0).sum())
    final_extreme_volume = len(
        extremes[extremes["extreme_volume_spike_flag"].fillna(False)]
    )
    final_extreme_return = len(
        extremes[extremes["extreme_return_flag"].fillna(False)]
    )
    final_extreme_price = len(
        extremes[extremes["extreme_price_flag"].fillna(False)]
    )
    retained_classification = (
        extremes.loc[
            extremes["extreme_return_flag"].fillna(False),
            "extreme_return_classification",
        ]
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )

    macros = [
        _macro("FinalSecurities", snapshot["securities"]),
        _macro("FinalIssuers", snapshot["issuers"]),
        _macro("FinalObservations", snapshot["observations"]),
        _macro("FilteredObservations", int(quality.loc[quality.metric.eq("Validated stock-day observations"), "original_liquidity_filtered"].iloc[0])),
        _macro("FergExcluded", snapshot["ferg_excluded"]),
        _macro("DkngExcluded", snapshot["dkng_excluded"]),
        _macro("FinalZeroVolumeRows", snapshot["zero_volume_rows"]),
        _macro("FinalZeroVolumeSecurities", snapshot["zero_volume_securities"]),
        _macro("FinalZeroVolumeEpisodes", snapshot["zero_volume_episodes"]),
        _macro("FinalZeroVolumeLongest", snapshot["longest_zero_volume_episode"]),
        _macro("OriginalZeroVolumeRows", int(quality.loc[quality.metric.eq("Zero-volume observations"), "original_liquidity_filtered"].iloc[0])),
        _macro("FinalExtremeVolumeRows", final_extreme_volume),
        _macro("FinalExtremeVolumeSecurities", snapshot["extreme_volume_securities"]),
        _macro("FinalExtremeVolumeEpisodes", snapshot["extreme_volume_episodes"]),
        _macro(
            "FinalExtremeVolumeMaxShares",
            f"{snapshot['extreme_volume_max_share_volume']:,.0f}",
        ),
        _macro(
            "FinalExtremeVolumeMaxDollar",
            f"{snapshot['extreme_volume_max_daily_dollar_volume']:,.2f}",
        ),
        _macro("FinalExtremeReturnRows", final_extreme_return),
        _macro("FinalExtremeReturnSecurities", snapshot["extreme_return_securities"]),
        _macro("FinalExtremeReturnPositive", snapshot["extreme_return_positive"]),
        _macro("FinalExtremeReturnNegative", snapshot["extreme_return_negative"]),
        _macro("FinalExtremeReturnEpisodes", snapshot["extreme_return_episodes"]),
        _macro(
            "RetainedExtremeMarketMoves",
            retained_classification.get("retained_extreme_market_move", 0),
        ),
        _macro(
            "SameDayDividendAdjustmentCases",
            retained_classification.get("same_day_dividend_adjustment", 0),
        ),
        _macro(
            "CorporateActionExtremeReturns",
            snapshot["corporate_action_related_extreme_returns"],
        ),
        _macro(
            "OverlappingVolumeReturnEpisodes",
            snapshot["overlapping_extreme_volume_return_episodes"],
        ),
        _macro("UniverseSecurities", 4874),
        _macro("AcquiredSecurities", 3239),
        _macro("SelectedSecurities", snapshot["securities"]),
    ]
    (generated / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    policy_table = policy[
        [
            "ticker",
            "eligible_from",
            "eligible_to",
            "policy_type",
            "verification_status",
            "decision_date",
        ]
    ].copy()
    policy_table.columns = [
        "Ticker",
        "Eligible from",
        "Eligible to",
        "Policy type",
        "Verification",
        "Decision date",
    ]
    policy_table["Eligible to"] = policy_table["Eligible to"].replace("", "open")
    policy_table["Verification"] = policy_table["Verification"].replace(
        {
            "verified_sec_filing_record_and_market_transition": (
                "Verified SEC filing record"
            )
        }
    )
    _table(
        policy_table,
        generated,
        "table_eligibility_policy.tex",
        "lrrp{0.22\\linewidth}p{0.25\\linewidth}r",
    )

    exclusion_table = (
        exclusions.assign(year=pd.to_datetime(exclusions["date"]).dt.year)
        .groupby("ticker", sort=True)
        .agg(
            excluded_observations=("date", "size"),
            first_excluded_date=("date", "min"),
            last_excluded_date=("date", "max"),
        )
        .reset_index()
    )
    retention_rows = []
    for ticker in ("DKNG", "FERG"):
        retained = final_daily.loc[
            final_daily["security_id"].astype(str).eq(
                policy.loc[policy["ticker"].eq(ticker), "security_id"].iloc[0]
            )
        ]
        retention_rows.append(
            {
                "ticker": ticker,
                "retained_observations": len(retained),
                "first_retained_date": retained["date"].min(),
                "last_retained_date": retained["date"].max(),
            }
        )
    exclusion_table = exclusion_table.merge(
        pd.DataFrame(retention_rows), on="ticker", how="left"
    )[
        [
            "ticker",
            "excluded_observations",
            "first_excluded_date",
            "last_excluded_date",
            "retained_observations",
            "first_retained_date",
            "last_retained_date",
        ]
    ]
    exclusion_table.columns = [
        "Ticker",
        "Excluded",
        "First excluded",
        "Last excluded",
        "Retained",
        "First retained",
        "Last retained",
    ]
    for column in (
        "First excluded",
        "Last excluded",
        "First retained",
        "Last retained",
    ):
        exclusion_table[column] = pd.to_datetime(
            exclusion_table[column]
        ).dt.strftime("%Y-%m-%d")
    _table(exclusion_table, generated, "table_historical_exclusions.tex", "lrrrrrr")

    final_daily_year = (
        final_daily.assign(year=pd.to_datetime(final_daily["date"]).dt.year)
        .groupby("year", sort=True)
        .agg(
            observations=("date", "size"),
            securities=("security_id", "nunique"),
            zero_volume=("volume", lambda values: int(values.eq(0).sum())),
        )
        .reset_index()
    )
    final_daily_year.columns = [
        "Year",
        "Observations",
        "Securities",
        "Zero volume",
    ]
    _table(final_daily_year, generated, "table_final_years.tex", "rrrr")

    zero_table = zero_by_security[
        [
            "ticker",
            "zero_volume_rows",
            "first_zero_date",
            "last_zero_date",
            "zero_volume_episodes",
            "classification",
        ]
    ].copy()
    zero_table.columns = [
        "Ticker",
        "Zero-volume rows",
        "First zero",
        "Last zero",
        "Episodes",
        "Classification",
    ]
    _table(
        zero_table,
        generated,
        "table_remaining_zero.tex",
        "lrrrrp{0.35\\linewidth}",
    )

    extreme_table = pd.DataFrame(
        [
            (
                "Extreme volume",
                final_extreme_volume,
                snapshot["extreme_volume_securities"],
                snapshot["extreme_volume_episodes"],
            ),
            (
                "Extreme adjusted price",
                final_extreme_price,
                snapshot["extreme_price_securities"],
                0,
            ),
            (
                "Extreme diagnostic return",
                final_extreme_return,
                snapshot["extreme_return_securities"],
                snapshot["extreme_return_episodes"],
            ),
        ],
        columns=["Category", "Observations", "Securities", "Strict episodes"],
    )
    _table(extreme_table, generated, "table_extreme_summary.tex", "lrrr")

    comparison = quality.copy()
    comparison["original_acquired_per_100k"] = comparison[
        "original_acquired_per_100k"
    ].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    comparison["original_liquidity_filtered_per_100k"] = comparison[
        "original_liquidity_filtered_per_100k"
    ].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    comparison["final_research_dataset_per_100k"] = comparison[
        "final_research_dataset_per_100k"
    ].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    comparison = comparison[
        [
            "metric",
            "original_acquired",
            "original_liquidity_filtered",
            "final_research_dataset",
            "original_acquired_per_100k",
            "original_liquidity_filtered_per_100k",
            "final_research_dataset_per_100k",
        ]
    ]
    comparison.columns = [
        "Metric",
        "Acquired",
        "Filtered",
        "Final",
        "Acq./100k",
        "Filt./100k",
        "Final/100k",
    ]
    _table(
        comparison,
        generated,
        "table_quality_comparison.tex",
        "p{0.32\\linewidth}rrrrrr",
    )

    treatments = pd.DataFrame(cleaning["treatments"])
    treatments.columns = ["Issue", "Final treatment"]
    _table(
        treatments,
        generated,
        "table_treatments.tex",
        "p{0.42\\linewidth}p{0.52\\linewidth}",
    )

    _figures(
        final_daily=final_daily,
        exclusions=exclusions,
        zero_by_security=zero_by_security,
        summary=quality,
        output_dir=figure_dir,
    )
    statistics = {
        "generated_at_utc": cleaning["generated_at_utc"],
        "snapshot": snapshot,
        "quality_summary": quality.to_dict("records"),
        "exclusions": exclusion_table.to_dict("records"),
        "remaining_zero_by_security": zero_table.to_dict("records"),
        "remaining_extreme_classifications": retained_classification,
        "conditional_accuracy": {},
        "protected_directory_hashes_sha256": cleaning[
            "protected_directory_hashes_sha256"
        ],
        "source_data_modified": False,
    }
    atomic_write_json(statistics, generated / "report_statistics.json")
    _write_report_source(
        report_root=report_root,
        snapshot=snapshot,
        hashes=cleaning["protected_directory_hashes_sha256"],
    )
    return {
        "final_observations": len(final_daily),
        "ferg_excluded": snapshot["ferg_excluded"],
        "dkng_excluded": snapshot["dkng_excluded"],
        "zero_volume_rows": final_zero,
        "extreme_volume_rows": final_extreme_volume,
        "extreme_return_rows": final_extreme_return,
        "extreme_price_rows": final_extreme_price,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "historical_eligibility.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
