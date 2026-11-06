# Missing-security investigation

## Observed cohort

The manifest contains one result for each of the 4,874 verified securities:
3,239 successful histories and 1,635 securities with no usable Yahoo rows. The
unavailable share is therefore 33.55%. Every recorded final failure is a Yahoo
symbol-history failure; there are no final transport or rate-limit errors in
this cohort.

This is an acquisition outcome, not a delisting finding. The security master
currently has no populated listing or delisting dates for any unavailable
security, so the investigation does not invent those events.

## Evidence categories

| Evidence category | Securities | Share of unavailable |
|---|---:|---:|
| No current SEC ticker match after Yahoo symbol miss | 1,243 | 76.02% |
| SEC-backed later ticker, old symbol miss | 379 | 23.18% |
| Current SEC ticker exists, but Yahoo returned no history | 13 | 0.80% |

The first category does not establish that the security was never listed or had
delisted. It means only that the requested symbol returned no history and the
current SEC ticker reference has no exact match.

## Descriptive cohort differences

Unavailable histories are not evenly distributed:

- Nasdaq: 1,146 of 3,038 unavailable (37.72%).
- NYSE: 454 of 1,729 unavailable (26.26%).
- NYSE American: 35 of 107 unavailable (32.71%).
- Common-equity classes: 531 of 986 unavailable (53.85%).
- Common stock: 1,043 of 3,755 unavailable (27.78%).
- Securities with a current SEC ticker match: 13 of 3,113 unavailable
  (0.42%).
- Securities without a current exact SEC ticker match: 1,622 of 1,761
  unavailable (92.11%).
- Securities with SEC-evidenced later ticker changes: 379 of 391 unavailable
  (96.93%).

These are descriptive differences with explicit denominators. The universe is
not a random sample of the unknown complete listed population, so no claim of
statistical representativeness follows.

## Targeted recovery assessment

The SEC-derived ticker-change strings identify 292 unavailable securities with
one unambiguous later symbol candidate. Another 87 have multiple candidate
symbols, and 1,256 have no later-symbol evidence. None of these aliases was
admitted automatically.

A small probe of 12 unambiguous alias candidates found vendor history for all
12. Eleven begin before the 3 January 2023 reference date, while one begins
after it. The probe demonstrates that targeted recovery is technically
feasible, but it does not prove that pre-change Yahoo history belongs to the
same security. Probe data are stored under
`data/investigation/missing/targeted_probe/` and remain outside the validated
research base.

One preferred-security-looking alias was excluded before the admitted probe
sample. This illustrates why a ticker found under the same SEC CIK is not
sufficient by itself: a CIK may have multiple security classes.

## Implications

The dominant missingness mechanism is historical symbol availability in the
vendor namespace. Because securities with ticker changes or without current
ticker matches are far less likely to be acquired, the available panel can
underrepresent renamed, delisted, merged, or otherwise removed securities.
That creates selection and survivorship risk even though no failed security is
silently removed from the manifest.

Further targeted recovery should use an explicit identity table that records
the alias, change date, share class, and continuity evidence. The next research
stage should report sensitivity to the unavailable cohort rather than treating
the Yahoo panel as representative.

## Reproduction

```bash
conda activate dsa5205_proj2
python scripts/run_missing_analysis.py
python scripts/run_missing_analysis.py --probe-limit 12
python -m pytest -q tests/test_missing_analysis.py
```
