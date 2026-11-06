# Secondary EODHD public-demo sample

The Yahoo Finance smoke test was blocked by Yahoo's regional access notice and
provider rate limiting. To exercise the market-data quality pipeline on real
observations without pretending that coverage is complete, the project includes
a separately labelled EODHD public-demo sample.

## Scope and interpretation

- The sample is limited and is not a replacement for the failed full-universe
  Yahoo acquisition.
- Selection is deterministic across exchange, security type, and whether a
  current SEC ticker matches the 2023 ticker. No beta, return, volume, or
  strategy characteristic is used.
- Public-demo credentials can restrict or return no data for individual
  securities. Those responses are recorded as failures, not silently omitted.
- EODHD is a secondary source for pipeline validation and limited descriptive
  evidence only. It must not be described as the full historical universe.

## Files

Raw API CSV/JSON response bodies are preserved under
`data/raw/eodhd_demo/raw_responses/`. Normalized daily Parquet files under
`data/raw/eodhd_demo/daily/` use the common validation schema. Acquisition
results and failure reasons are in:

- `data/metadata/eodhd_demo_manifest.csv`
- `data/metadata/eodhd_demo_failures.csv`
- `data/metadata/eodhd_demo_provenance.json`

Dividend values use the reported unadjusted cash amount so raw-price continuity
checks remain comparable to the unadjusted Close field. Stock-split factors are
parsed from EODHD's ratio string. Reported share volume remains vendor-reported;
the sample does not assert a split-normalized volume convention.

## Reproduction

```bash
conda activate dsa5205_proj2
python scripts/download_eodhd_sample.py
python scripts/validate_data.py \
  --config configs/acquisition.yaml \
  --raw-dir data/raw/eodhd_demo/daily \
  --master data/universe/security_master.parquet \
  --output-dir data/validated \
  --metadata-dir data/metadata
```
