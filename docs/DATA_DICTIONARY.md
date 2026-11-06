# Strategy input contract

Location: `data_current/adtv100m_2023_eligibility_v3_ohlc/`.
The SHA-256 manifest defines the historical snapshot used by the frozen model.

| Artifact / field | Meaning |
|---|---|
| `selected_equity_daily.parquet` | Unique `(security_id, date)` daily observations |
| `date` | Timezone-naive XNYS session timestamp |
| `security_id` | Stable project security key; join identity before ticker |
| `ticker` | Display/provider symbol, not a permanent identity key |
| `open`, `high`, `low`, `close` | Vendor raw OHLC in the recorded adjustment convention |
| `adj_close` | Adjusted close used for portfolio marks and fills |
| `volume` | Reported share volume; zero remains distinct from missing |
| `dividends`, `stock_splits` | Vendor corporate-action fields |
| `vendor_repair_flag` | Vendor repair indicator |
| `pct_return` | Upstream placeholder, retained null; strategy derives its own returns |
| `ohlc_source*`, `ohlc_adjustment_convention` | Local OHLC provenance and convention |
| `selected_security_master.parquet` | One row per security, including `security_id` and `cik` |
| `cik` | SEC issuer key, used to aggregate issuer exposure |
| `dataset_metadata.json` | Snapshot counts, quality and enrichment status |
| `ohlc_source_provenance.json` | Historical source hashes and exclusions |
| Remaining CSV/Markdown audit files | Coverage, exclusions, removed rows and schema changes |

Raw and adjusted fields must not be interchanged. The event-day low is adjusted
using `adj_close / close`; CLV uses mutually consistent raw OHLC. Entry liquidity
uses lagged raw close times reported volume. The model's date predicate is
applied before Arrow materialises market observations.

Daily backtest outputs contain `date`, total `nav`, active and core component
indices, portfolio return, cash and equity exposure. `active_nav` and `core_nav`
are separately normalised indices; reconstruct the combined NAV using the
configured initial weights rather than adding the two columns directly.
