# Liquidity-filter contracts

## Measurement window

The fixed liquidity cohort uses the final 20 XNYS sessions ending
`2022-12-30`. A security is eligible only when all 20 sessions have a finite
positive nominal close and finite nonnegative reported share volume. A genuine
zero-volume session is valid and contributes zero dollar volume. Missing values
and missing sessions are never filled or replaced with zero.

## Price-volume compatibility

Yahoo's nominal `Close` is split-back-adjusted across the full retrieved
history, while its reported `Volume` is interpreted as share volume on the
contemporaneous trading-date share basis. For each measurement-window row:

`compatible_close = nominal_close * product(split factors dated after row date and <= 2025-12-31)`

`daily_dollar_volume = compatible_close * reported_volume`

The split restoration uses future corporate-action metadata solely to interpret
historical vendor adjustment conventions. It does not use future prices,
returns, or volumes. The methodology and unresolved assumptions are recorded in
`filter_metadata.json`.

## Selection statuses

- `selected`: complete valid window and ADTV20 at least USD 100 million.
- `excluded_below_threshold`: complete valid window and ADTV20 below the threshold.
- `insufficient_liquidity_window_data`: fewer than 20 valid specified sessions.
- `unresolved_price_volume_issue`: the source fields cannot safely form dollar
  volume under the documented convention.
- `not_acquired`: no Yahoo history entered the fixed screen.

## Preservation

The original security master, Yahoo checkpoints, validated base, diagnostic
tables, and issue registers are immutable inputs. Filtered outputs live only
under `data/filtered/adtv100m_2023/`.
