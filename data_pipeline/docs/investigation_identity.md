# Security identity and temporal coverage investigation

## Provenance

- Yahoo manifest: 3239 successful and 1635 failed of 4874.
- Yahoo raw files: 3239.
- Yahoo raw observations: 7,128,162.
- Validated base: 7,086,540 rows and 3,239 securities.
- Base IDs outside successful Yahoo IDs: 0.
- EODHD demo files are outside the Yahoo raw directory; 7,542 secondary rows are not part of the Yahoo base.

The prior abstract was hardcoded and retained obsolete failure text after the body-generation logic was updated. Generated statistics and the Yahoo manifest are authoritative; the abstract must consume those statistics through macros rather than repeat old prose.

## Checkpoint identity

- Checkpoints audited: 3,239.
- Identity-valid: 3,239.
- Identity-invalid or quarantined: 0.
- Checkpoints with duplicate dates: 0.
- Checkpoints containing all-NaN price placeholders: 37 (41,617 rows).
- Exact cross-security history candidates: 0.

The manifest ticker, Parquet ticker, security ID, 2023 ticker, and provider were compared for every checkpoint. All-NaN rows fail the price-observation test and never enter the research base.

## Temporal coverage

Annual counts distinguish any positive-price vendor row, positive-volume trading records, and cumulatively observed securities. The 2025 equality between successful histories and positive-price records is not by itself evidence of contamination.
Securities with at least 10 trailing market sessions after their last positive-volume trade and at least 10 trailing unchanged zero-volume rows: 12.

See `data/investigation/identity/` for the checkpoint-level tables and machine-readable findings.
