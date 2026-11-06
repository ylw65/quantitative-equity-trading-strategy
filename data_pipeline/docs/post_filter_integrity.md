# Post-filter integrity assessment

The filtered cohort contains 488 securities representing
486 distinct SEC issuers. The retained base
contains 1,187,986 validated stock-day rows.

Raw selected-cohort reconciliation finds 1,190,496 raw
rows, 2,510 raw rows excluded from the
validated base, and 2,510 all-NaN
price placeholders. Missing adjusted-close and volume rows are retained in the
audit archive and are never filled with zero.

The selected cohort has 49
securities whose observed history does not span the full calendar. Because the
master lacks verified listing and delisting dates, leading and trailing periods
remain unknown rather than classified as structural.

The identity review flag overlaps 2
selected securities: ABNB, AFRM.
Their current action is `base_rows_only`, not confirmed cross-ticker
contamination; the evidence file is retained in `identity_review_candidates.csv`.

Protected source hashes were unchanged after the audit: True.
