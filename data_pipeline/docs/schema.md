# Shared data contracts

## Universe and security master

`universe_2023.csv` is the human-auditable reference-date universe.

`security_master.parquet` contains one row per eligible 3 January 2023 common
equity security. Required fields are:

- `security_id`: `SEC:CIK##########` when SEC CIK is available; otherwise a
  deterministic fallback identifier prefixed `TICKER2023:`.
- `ticker_2023`, `yahoo_ticker`, `company_name`, `exchange`, `security_type`
- `permanent_id_type`, `permanent_id`
- `listing_date`, `delisting_date`, `subsequent_ticker_changes`
- `included`, `exclusion_reason`, `source`, `source_confidence`
- `reference_date_verified`

Empty values mean unknown. They are not evidence that an event did not occur.
Ticker symbols are identifiers, not permanent security identities.

## Raw Yahoo observations

Each security/ticker is stored independently under `data/raw/yahoo/`. Raw files
retain vendor values before validation or adjustment. Expected columns:

`date, open, high, low, close, adj_close, volume, dividends, stock_splits`.

The raw file may also retain `ticker`, `security_id`, `retrieved_at_utc`,
`provider`, `download_id`, and provider-specific source columns.

## Validated research-facing base

The required long-form base columns are:

`date, security_id, ticker, volume, adj_close` and reserved `pct_return`.

`pct_return` remains null in the finalized base dataset for this phase.
Diagnostic returns are stored only in a clearly labeled diagnostic table.

Validation must never zero-fill prices or volume, silently delete extreme
returns, or collapse distinct securities solely because they reused a ticker.
