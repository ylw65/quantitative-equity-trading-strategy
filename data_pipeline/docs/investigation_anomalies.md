# Extreme-price, return, and corporate-action investigation

## Scope and reproducibility

This investigation uses the preserved Yahoo checkpoints and the validated
diagnostic and quality tables. It does not alter raw files, `base.parquet`, or
the finalized null `pct_return` field. The reproducible entry point is:

```bash
python scripts/run_anomaly_investigation.py
```

Outputs are written under `data/investigation/anomalies/`, with copies of
reporting tables under `reports/tables/investigation/`. The raw SEC evidence
retrieved for the representative DHR and KDP events is also retained locally.

## Extreme diagnostic returns

The independent chronological calculation reproduces exactly **5,409** rows
whose adjusted return exceeds the absolute 50 percent threshold. The
classification precedence is: nonpositive/invalid denominator handling,
near-zero denominator, explicit simultaneous or nearby corporate action,
vendor repair flag, unresolved large discontinuity, then retained market move.

| Evidence category | Rows |
|---|---:|
| Retained extreme market move | 4,013 |
| Near-zero denominator | 1,244 |
| Split adjustment within five sessions | 89 |
| Same-day split adjustment | 24 |
| Unresolved extreme discontinuity | 20 |
| Vendor-repaired price discontinuity | 17 |
| Same-day dividend adjustment | 2 |

**Observed.** VAXX dominates the upper tail. Its previous adjusted close was
USD 0.0002 before a 2025-03-26 adjusted close of USD 5.00, producing a
24,999.0 percent diagnostic return with a vendor repair flag. CLVR, SNAX and
XELA exhibit the same sub-cent denominator pattern.

**Tests performed.** Previous/current raw Close, Adjusted Close, Volume,
Dividend, Stock Split, and Yahoo repair flags were checked on the event rows.
Returns were recalculated only when both chronological adjusted closes were
positive.

**Conclusion.** At least 1,244 extremes are mechanically inflated or unstable
because the previous adjusted price is at or below USD 0.01. They are not safe
to interpret as ordinary percentage returns. A further 20 lack a direct action
or repair explanation, and 17 carry a vendor repair flag.

**Uncertainty.** The evidence can identify economically fragile denominators and
provider repairs but cannot establish a tradable quote from a sub-cent price
without an independent consolidated-tape source.

**Recommendation.** Preserve all raw observations, but quarantine the listed
event windows from return-based diagnostics. This is not a strategy eligibility
rule.

## Extreme adjusted-price levels

Seventy-three securities exceed the investigation thresholds for adjusted-price
levels. Thirty-four are consistent with a history containing reverse splits;
thirty-nine contain sub-cent adjusted observations. WHLR is the leading level
case: its checkpoint contains 12 reverse split events, a cumulative split
product of approximately `2.07e-9`, a maximum adjusted close near
`1.73e13`, and about 88.7 percent zero-volume rows. This pattern is
arithmetically consistent with repeated reverse-split back-adjustment rather
than a contemporaneous USD 17 trillion share price.

These are not classified as valid price levels for feature engineering merely
because the arithmetic reconciles. They are retained for audit and recommended
for quarantine from unlogged level transformations.

## Split correction

The three split-continuity flags produced by the earlier validation formula
were reproduced and all coincide with a dividend/distribution on the same date:

| Security | Event | Split | Distribution | Classification |
|---|---|---:|---:|---|
| DHR | 2016-07-05 | 1.319 | 24.56 | simultaneous separation/distribution |
| CMCT | 2019-09-03 | 0.3333 | 102,941.18 | simultaneous reverse split/distribution |
| CRESY | 2025-11-28 | 1.025 | 0.624 | simultaneous split/distribution |

The original single-action test is therefore not applicable. The current
vendor-aware validator excludes combined-action dates from the simple
split-continuity population and records them separately. This leaves **zero
residual simple split-continuity failures** among these three events. Raw split
metadata remains unchanged.

Danaher's 2016-07-08 SEC Form 8-K independently documents agreements for the
separation of Fortive and distribution of its shares. Source:
`data/investigation/anomalies/sec_DHR_2016-07-08_8k.htm`.

## Dividend/distribution reconciliation

The current vendor-aware pass reconciles **31 dividend/distribution flags**:

| Classification | Events |
|---|---:|
| Probable special distribution or spin-off | 21 |
| Large distribution or adjustment artifact | 8 |
| Ordinary dividend adjustment mismatch | 2 |

The earlier 33-flag count included two simultaneous split/distribution events.
Those events are now recorded once in the split reconciliation above, so the
current reports do not double-count them.

Twenty-eight remain partially resolved because Yahoo's single `Dividends`
field does not distinguish cash dividends, special distributions, spin-offs,
and reorganizations. Two ordinary-dividend cases remain unresolved. KDP's
2018-07-10 event is independently verified by its 2018-07-09 SEC Form 8-K,
which reports a special cash dividend of USD 103.75 per share. Source:
`data/investigation/anomalies/sec_KDP_2018-07-09_8k.htm`.

DHR's distribution and KDP's special dividend are resolved as event types, but
their exact vendor adjustment mathematics remains a separate accounting
question.

## Quarantine register

The investigation produced **1,386 quarantine-candidate rows or windows across
133 securities**. These are recommendations scoped to return diagnostics or
price-level transformations. They do not remove raw rows, modify the research
base, or establish final strategy eligibility.

## Readiness

- Descriptive coverage and raw-field analysis: ready with cited exclusions.
- Return-based diagnostics: requires filtering or explicit quarantine of
  near-zero denominators and unresolved repaired discontinuities.
- Level-based price transformations: requires quarantine of the 73 anomalous
  price-level securities.
- Strategy feature engineering or modeling: not ready until analysis eligibility
  is approved and chronological splitting is performed in a later phase.
