# Data-quality methodology

## Scope and preservation

Validation starts from per-security raw vendor output (the primary Yahoo route
or the separately labelled secondary sample) and never overwrites it.
Prices and volumes are not zero-filled. Extreme observations are retained. The
research-facing `base.parquet` contains valid observations with `date`,
`security_id`, `ticker`, `volume`, `adj_close`, and a reserved `pct_return`
column whose values remain null in this phase. Missing, zero, and negative
price placeholders remain in the raw and standardized audit tables but are not
admitted to this research-facing base.

`standardized_raw.parquet` preserves the acquired vendor fields with normalized
names and types. The source raw file remains the authoritative audit copy,
including unexpected extra fields. If a security-date key is duplicated, every
raw row remains in the raw/audit data, the issue register records the count, and
the first deterministic vendor row is used in the unique-key base.

## Calendar and structural missingness

Expected sessions come from `exchange_calendars` using XNYS. Non-session vendor
rows are flagged. Weekend and holiday dates are neither inserted nor counted as
stock-level gaps. `calendar_summary.csv` separately reports weekend days and
weekday exchange holidays so these structural calendar absences are not
confused with vendor gaps. Missing sessions between each security's first and last
observed dates are reported as unobserved sessions; the cause is not inferred.

Known listing and delisting dates from the security master distinguish
structural pre-listing/post-delisting absence where available. Where listing
status is unknown, the validator labels the condition unverified rather than
asserting an IPO, suspension, or delisting. An end date before 31 December 2025
without a verified delisting remains an unresolved history ending.

`verified_active_missing_sessions` is computed only when both listing and
delisting dates are verified. The broader observed-interval missing count is an
audit metric, not a claim that the full interval was an active listing.

## Diagnostic returns

The validator computes a temporary adjusted-close total return solely to flag
discontinuities. It is stored as `diagnostic_pct_return` in
`diagnostic_returns.parquet`; it is not written into the final base
`pct_return`. No clipping, winsorization, labels, signals, fitted
transformations, or strategy features are produced.

The default review threshold is absolute daily diagnostic return of 50%. A
crossing can be a genuine market shock or a bad observation and is not deleted.
Open/high/low consistency, zero or negative prices, and missing values receive
separate flags. On dates without a reported split or dividend, raw-close and
adjusted-close gross returns should agree; disagreement beyond one percentage
point creates a separate adjustment-conflict review flag.

## Trading volume

Missing, negative, and zero share volumes are distinguished. An extreme-volume
diagnostic uses only a preceding rolling median internally and writes a Boolean
review flag; no ADV20, relative-volume series, or trading feature is exported.

Reported volume is retained in vendor share units. The data do not establish
that either vendor normalized historical volume for splits. Pre/post-split
volume must not be compared as if it were on a constant share basis without an
explicit future conversion. Reported share volume is also conceptually distinct
from dollar volume.

## Corporate actions

Split and dividend events reported by Yahoo are retained. Yahoo's nominal Close
is historically back-adjusted across splits even when `auto_adjust=False`, so
the Yahoo audit checks that nominal and adjusted closes imply approximately
unit continuity at split dates while preserving the split multiplier as
metadata. An unadjusted close series from another provider is checked against
the event multiplier instead. For dividends, adjusted-price gross return is
compared with the raw-close total return including the reported cash dividend.
Dates carrying both a split and a distribution are excluded from either
single-action continuity test and recorded as simultaneous corporate actions.
Dates with both a split and a cash/distribution event are excluded from the
simple split and dividend formulas and logged as simultaneous corporate
actions. A single-action continuity formula is not valid for those dates.

These are screening diagnostics: overlapping market movement, event timing, and
vendor methodology can produce flags. A split-date volume level change near the
split factor is recorded as an informational observation, not an error
determination.

## Issue register and reporting

`data_issues.csv` groups repeated row-level problems by security/type while
retaining affected counts and date bounds. Severity reflects review priority, not
an automatic exclusion decision. Status and resolution retain what was checked
or what remains unresolved. Acquisition failures are carried from
`download_manifest.csv` and `download_failures.csv`.

Per-security and per-year tables report observation counts, missing row values,
calendar gaps within observed intervals, extreme diagnostics, action counts,
history status, and the fewer-than-252-valid-session flag. Coverage is never
described as complete when the historical universe denominator is unknown.
