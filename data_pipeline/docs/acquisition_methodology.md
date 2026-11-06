# Yahoo Finance acquisition methodology

## Purpose and boundary

This stage acquires raw daily market observations for the verified 3 January
2023 historical universe. It does not compute a finalized return field, filter
stocks by trading characteristics, create strategy features, split the sample,
or run models/backtests.

`yfinance` is called with:

- start `2016-01-01`, end `2026-01-01` exclusive (31 December 2025 inclusive);
- `auto_adjust=False`, preserving raw Close and vendor Adj Close separately;
- `actions=True`, preserving dividends and stock splits;
- batched requests (default 50 tickers), no internal yfinance threads, a
  conservative pause, and bounded retries.

## Identity and checkpoint layout

The security master supplies `security_id`, the 2023 ticker, and the
Yahoo-compatible ticker separately. Yahoo history may refer to a subsequent
ticker if a symbol changed. Such mappings remain explicit in the security
master and are not inferred from ticker similarity.

Each security/ticker is written to its own Parquet checkpoint below
`data/raw/yahoo/`. Files are atomically replaced after successful retrieval.
The raw frame retains all vendor chart fields supplied by yfinance, including:

`date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `dividends`,
and `stock_splits`. It also records `ticker`, `ticker_2023`, `security_id`,
provider, download ID, and UTC retrieval time. Unknown provider fields are
retained under normalized names.

Yahoo's output can vary between a simple column index and a MultiIndex. The
pipeline extracts each ticker explicitly and rejects empty/ambiguous frames
without inventing observations. Yahoo sometimes returns a date-only grid filled
entirely with `NaN` for an absent or delisted symbol. These placeholders are
quarantined under `data/raw/yahoo_invalid/` and recorded as failures rather than
as market observations.

## Isolation, retry, and resume behavior

1. A batch request is attempted up to the configured retry count.
2. Successful symbols are checkpointed immediately; missing symbols do not
   invalidate the rest of the batch.
3. Missing symbols are then attempted individually.
4. A provider-wide rate limit stops further requests immediately to avoid
   increasing request pressure.
5. Reruns skip a security only if its manifest entry is successful/partial and
   its raw checkpoint exists. If a process stopped after a raw checkpoint but
   before a manifest flush, the checkpoint is recovered from Parquet metadata.

`data/metadata/download_manifest.csv` contains the latest result for each
ticker. `data/metadata/download_failures.csv` contains failed and structurally
partial downloads. A `history_status` value such as
`starts_after_requested_start` records observed period coverage; it does not by
itself imply a data error because an IPO may legitimately postdate 2016.

## Provenance

Every run writes package versions, Python/platform details, date range,
acquisition parameters, retained fields, and research constraints to
`data/metadata/acquisition_run_provenance.json`. The per-row manifest records
the run ID and retrieval/completion timestamps.

## Full-run finding

After network access was restored, the pipeline attempted all 4,874 verified
securities. It produced 3,239 successful Yahoo histories and 1,635 unavailable
or empty histories, totaling 7,128,162 raw rows. The implementation preserves
successful checkpoints, reproduces deterministic failures, and permits later
resumption without deleting prior failures.

A temporary Yahoo TLS outage was observed during acquisition. A current SEC
ticker match is therefore never classified as permanently delisted after a
transport failure; those symbols remain retryable. Missing and unavailable
symbols are retained in the manifest rather than silently removed.
