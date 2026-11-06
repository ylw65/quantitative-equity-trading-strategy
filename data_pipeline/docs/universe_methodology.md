# Historical US common-equity universe methodology

## Objective and reference date

The target is a reproducible, best-verifiable reconstruction of common equities
listed on NYSE, Nasdaq, or NYSE American on 3 January 2023. The objective is a
historical reference universe, not a current ticker directory.

No statement in the outputs treats the true historical listing census as known.
The deliverable is a verified partial universe with explicit uncertainty.

## Source hierarchy

1. **SEC Q4 2022 and Q1 2023 quarterly filing indexes.** These identify US
   registrants that filed annual or quarterly reports around the reference date.
   For each CIK, the build chooses the latest primary filing on or before
   3 January 2023 and otherwise the earliest primary filing afterward.
2. **Section 12(b) cover data in the selected filing.** Inline XBRL fields
   `dei:TradingSymbol`, `dei:SecurityExchangeName`, and
   `dei:Security12bTitle` provide the filing-time ticker, exchange, and security
   title. Only a bounded compressed range at the start of the filing is read;
   extracted metadata are cached independently from the source filing.
3. **SEC current ticker/exchange reference.** This is used to identify apparent
   subsequent ticker changes and to flag unresolved current-listing
   supplements. It is never treated as a complete January 2023 universe.

Every source URL, cache path, SHA-256 digest, access timestamp, parameter, and
software version is recorded in
`data/metadata/universe_source_provenance.json`.

## Scope classification

Common stock, common equity classes, ordinary shares of non-ADR foreign issuers,
and REIT common shares are eligible when the filing cover verifies an eligible
exchange and security type. ADRs, funds and ETFs, preferred shares, warrants,
rights, units, debt, notes, and narrowly defined royalty/income/unit trusts are
excluded with the observed title retained.

Ambiguous titles are excluded from the verified security master and entered in
`data/universe/universe_issues.csv`. The build does not infer common-equity status
from a plausible company name alone.

## Reference-date verification

`reference_date_verified=true` means a latest pre-reference SEC primary filing
provided an eligible security title and exchange. This is strong listing
evidence, not a daily exchange official-list confirmation. `source_confidence`
distinguishes this case from a post-reference filing fallback.

The acquisition stage must further check whether a security has a market
observation on or immediately around 3 January 2023. No universe record is
declared free of IPO or listing-date error when only post-reference filings are
available.

## Identity and ticker policy

SEC CIK is the preferred permanent entity identifier. Ticker is retained as a
reference-date label and is not assumed permanent. When a CIK has multiple
covered common classes, the 2023 ticker is appended to the internal
`security_id` to keep security-level observations distinct. A CIK identifies an
issuer, while a SEC filing can cover multiple classes.

Apparent ticker reuse or two CIKs sharing one 2023 ticker is treated as a
high-severity identification issue. Ticker changes inferred by comparing the
2023 cover symbol with the current SEC reference are recorded, but intermediate
symbol histories are not invented.

SEC filings from operating subsidiaries and partnerships can repeat a parent
issuer's covered security. When the same reference ticker occurs more than once,
the audit universe retains every cover-page row, but the acquisition master
retains exactly one security. A current SEC ticker match outranks a subsidiary
cover row; otherwise the ranking favors the likely operating issuer over
partnership and limited-liability subsidiary forms. The loser rows are marked
excluded and both the duplicate resolution and underlying cross-CIK audit are
recorded as issues. Ambiguous mappings are not merged silently.

Ticker placeholders such as `n/a`, malformed symbols, and inline-XBRL footnote
artifacts are rejected. In particular, a foreign issuer's ordinary shares are
not admitted merely because a non-trading `n/a` cover row names a US exchange;
the separately listed ADR remains excluded under the security-type rule.

## Completeness

The accessible sources do not provide an official date-specific NYSE, Nasdaq,
and NYSE American common-equity roster for 3 January 2023. Consequently:

- coverage percentage remains unknown;
- SEC filer counts are reported as source denominators, not as the full listed
  common-equity denominator;
- download success and verified inclusion are reported separately;
- securities with failed cover extraction can be supplemented only as flagged,
  non-included candidates until independent evidence is obtained.

The likely bias is non-random rather than a simple survivor bias. Delisted firms
are partially retained through historical SEC filings, but firms with weak
inline-XBRL coverage, nonstandard foreign filings, or missing filing-index
representation are more likely to be absent. Any downstream analysis using this
universe must report that verified coverage remains partial.

## Reproduction

Set a descriptive SEC user agent and run:

```bash
export SEC_USER_AGENT='Academic project name and contact email'
conda activate dsa5205_proj2
python scripts/build_universe.py
```

The cover cache makes interrupted runs resumable. `--force` discards cached
source downloads and extraction results.
