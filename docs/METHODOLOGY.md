# Research design

## Data and eligibility

The acquisition study used SEC evidence to identify 4,874 verified common-equity
securities; 3,239 returned Yahoo histories. Universe completeness is unknown.
The validated base contained 7,086,540 stock-days. A fixed reference cohort
screened the final 20 XNYS sessions of 2022 at USD 100 million ADTV.

Cleaning retained raw observations and separated diagnostic returns from the
research panel. Historical eligibility removes pre-policy FERG and DKNG rows.
Identity review removes the verified wrong-issuer `B` series; the subsequent
OHLC stage excludes HUBB under a recorded manual policy. The final snapshot has
486 securities and 1,181,168 rows across 2016–2025. These full-history counts
are upstream audit facts, not permission to use future rows for modelling.
The backtest materialises only dates allowed by its stage.

## Signal and portfolio

A long event requires five-session return <= -6.5% and current volume divided
by the preceding 20-session mean in [1.5, 5]. The event score combines
cross-sectional volume and decline ranks equally. Entry rank is 20, retention
rank 40; entry also requires prior 20-session raw dollar volume >= USD 20m.

The 40% active sleeve is divided into five staggered accounts, each rebalancing
every five sessions. Exits include an event-low breach, a strong reverse event,
rank removal, invalid data or signal expiry. A signal can refresh; signal age
is not a fixed holding-period cap. Rejected entry segments remain in cash.
The 60% core buys equal-dollar positions initially and holds fixed shares.
No transfers restore the initial 60/40 split.

## ML and chronological controls

Seventeen price, volume, volatility, event and market-breadth features feed a
logistic regression (C=0.1). Imputation, 1%/99% clipping and scaling are fitted
only on previously completed trades. Labels include entry, interim resizing
and exit fees. Open segments are right-censored. Samples are weighted equally
by decision date. Each year's classifier uses labels completed before that
year; sigmoid calibration uses prior annual predictions with completed labels.

The admission threshold is the median of past calibrated predictions. The
frozen 2023 model uses 12,990 completed labels, latest 2022-12-30, and a
threshold of 0.43657775028548135. This is an estimated score threshold, not an
observed win rate. Models do not resize already accepted segments midstream.

## Execution and metrics

Shares are decided at close t and executed at close t+1. Sells execute first;
cash limits buys. Fractional shares and adjusted-close fills are research
approximations. The ledger holds units between rebalances, marks terminal
positions rather than liquidating, pays zero on cash, and charges 10/20/30 bps
per side. No leverage or shorting is used. Active single-security/issuer 5%
limits are target limits; price gaps can breach them after execution.

The headline return includes the passive core. Training uses the delivered
486-security cohort; primary 2023 evaluation uses 474 securities frozen from
the final 20 training sessions. This does not fix historical universe bias.
Sharpe, drawdown, average stock exposure and costs use the original frozen
implementation; nine reference scenarios verify the integrated edition.
