# Volume-quality investigation

The validated Yahoo panel contains 138,694 zero-volume
rows across 781 securities. Half of these
rows are concentrated in 63 securities.
41 securities meet the descriptive prolonged-case
screen of at least 50% zero-volume observations or a contiguous run of at
least 252 rows.

Zero volume is retained as zero, not converted to missing. It may represent
illiquidity, a suspension, an inactive vendor row, or another unresolved
condition; this audit does not infer which without additional evidence.

The panel contains 1,311 split events
with prior-session volume. Reported share volume is retained without
normalization. The event neighborhood mixes genuine trading, split mechanics,
and vendor conventions, so pre/post-split volume must not be interpreted as
constant-share activity without a later documented transformation.
