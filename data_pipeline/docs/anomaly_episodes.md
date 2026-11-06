# Filtered anomaly episode methodology

This directory contains a diagnostic event-period extraction from the fixed
ADTV-filtered cohort. The analysis does not modify vendor, validated, filtered,
or source issue-register files and initiates no downloads.

Episodes use strict adjacency on official XNYS sessions. Weekends and exchange
holidays are skipped; an intervening unflagged or absent exchange session ends
an episode. Each flagged stock-day belongs to one episode and one security.
Missing observations are preserved as null and are never converted to zero.

Required anomaly categories are zero volume, existing extreme-volume flags,
existing extreme adjusted-price evidence, and existing diagnostic returns
above the absolute 50 percent review threshold. Additional traceable labels
preserve OHLC consistency errors, quarantine windows, and the four selected
corporate-action reconciliation records. Quarantine and corporate-action labels
augment rather than replace the required anomaly categories.

Daily dollar volume restores the contemporaneous price basis by multiplying the
vendor nominal Close by future split factors strictly after each observation and
then multiplying by reported share volume. This follows the liquidity-screen
methodology and does not use future volume or returns.

Cumulative diagnostic return compounds only the flagged sessions in an episode.
It is null if any flagged session lacks a finite diagnostic return or implies a
non-positive return denominator. No return is imputed.

Review priority is technical rather than an exclusion decision. `P0` marks
episodes touching the December 2022 liquidity window. `P1` marks unresolved
price/return or corporate-action evidence, quarantine evidence, overlapping
anomaly types, multi-session zero volume, or an extreme-volume episode whose
maximum daily dollar volume reaches USD 100 million. `P2` covers remaining
corporate-action or anomaly reviews. `P3` covers isolated extreme returns that
the existing evidence classifies as retained market movements.

## Result

- Strict anomaly episodes: 533
- Affected securities: 134
- High-priority P0/P1 reviews: 354
- Validation checks passed: 10/10
- zero_volume: 317 episodes, 900 flagged stock-days
- extreme_volume: 205 episodes, 282 flagged stock-days
- extreme_price: 0 episodes, 0 flagged stock-days
- extreme_return: 41 episodes, 45 flagged stock-days
- ohlc_consistency: 1 episodes, 1 flagged stock-days
- raw_adjusted_conflict: 0 episodes, 0 flagged stock-days
- negative_volume: 0 episodes, 0 flagged stock-days
- non_session_date: 0 episodes, 0 flagged stock-days
- quarantine_candidate: 2 episodes, 4 flagged stock-days
- corporate_action_discrepancy: 4 episodes, 4 flagged stock-days

The context export contains five XNYS sessions before and after each episode,
clipped to 2016-01-04 through 2025-12-31. When context windows overlap, the same
stock-day appears once per referencing `period_id`; this is deliberate and
prevents context from changing the strict episode boundaries.

The review register leaves economic interpretation, external evidence, reviewer
conclusion, and treatment recommendation blank for human completion.
