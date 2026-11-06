# Reporting and exploratory-analysis plan

## Non-negotiable reporting rules

1. Every empirical number must be generated from raw, manifest, validated, or
   issue-register artifacts. Report prose may not introduce a hard-coded count.
2. Verified facts, interpretation, and unresolved hypotheses use distinct
   language. For example, "volume falls by the split factor" is an observation;
   "the vendor normalized volume" is a possible explanation until checked.
3. Missing values are not plotted or summarized as zero. Structural calendar
   absences, pre-IPO, post-delisting, suspension, vendor failure, and unknown
   gaps remain separate categories.
4. Extreme returns and volumes remain in the dataset and register. They may be
   described and flagged, not automatically removed, clipped, or winsorized.
5. Coverage percentages are emitted only when both numerator and defensible
   historical denominator are available. Otherwise emit `Unknown`.
6. Diagnostic returns are named "diagnostic return" in all figures, tables,
   filenames, and prose and are excluded from the research-facing feature set.
7. No figure or table may use current-listed-only membership as a proxy for the
   3 January 2023 universe.

## Required figure specifications

| Figure | Data contract | Required encoding | Caption must state |
|---|---|---|---|
| Listing-history length | `data_quality_by_security.csv` or security master | Histogram and/or ECDF of observed listing days | Observation window, unknown-date handling |
| Available by year | Validated observations | Count of distinct `security_id` with at least one valid adjusted close, by year | Whether membership is count-based or duration-weighted |
| Observations by year | Validated observations | Count of valid rows by year; structural missingness separately labelled | Calendar and reference-window boundaries |
| Missingness distribution | Per-security active-period audit | Histogram of missing-session rate, with unit and eligible denominator | Only verified active periods enter denominator |
| Adjusted-close distribution | Validated base | Positive values on logarithmic horizontal scale, common x-range | Pooling and repeated-security weighting |
| Volume distribution | Validated base | Positive reported share volume, logarithmic horizontal scale | Reported share volume, not dollar volume |
| Split audit | Raw and validated action-event windows | Paired adjusted price/return paths around representative events | Signed split semantics and volume normalization uncertainty |
| Representative histories | Selected eligible securities | Adjusted close and volume, one row per security | Selection method; illustrative not random |
| Extreme observations | Diagnostic-return table | Ranked/tail diagnostic and event flags | Not used as signal or final feature |
| Missingness heatmap (optional) | Security-by-year matrix | Missing rate by year for selected long-lived securities | Security selection rule |

Every figure is exported as PDF, using vector graphics, at least 8 pt font,
embedded fonts, readable grayscale lines, and a source-note line generated from
the artifact timestamp. PNG previews may be produced for QA but are not report
inputs.

## Required table specifications

| Table | Required columns |
|---|---|
| Universe by exchange | Exchange; included count; unresolved count; inclusion rule |
| Universe by security type | Security type; count; ADR/REIT status; inclusion rule |
| Download status | Attempted; success; partial; unavailable; resumed; failed; coverage status |
| Coverage by year | Year; securities; valid observations; active with data; unknown/unavailable |
| Missingness by year | Year; expected active sessions; missing adjusted close; missing volume; unexpected gaps |
| Gap examples | Security; start; end; missed sessions; likely class; evidence; status |
| Corporate-action counts | Year; securities with split; split events; dividend events; unresolved checks |
| Split audit | Security; date; ratio; pre/post diagnostic return; volume ratio; raw/adj consistency; status |
| Volume flags | Flag; observations; securities; severity; action |
| Extreme returns | Security; date; diagnostic return; prior/next close; split/dividend proximity; interpretation; status |
| Identity issues | CIK/ticker; issue; reuse/change evidence; severity; resolution |
| Issue summary | Severity; category; open; resolved; accepted limitation |
| Quality by security/year | Counts and rates of duplicate, price, volume, action, and missingness flags |

Tables use `booktabs`, thousands separators, explicit units, and sample-size
notes. Long tables repeat headers. Financial numbers are not presented without
their date range and observation basis.

## EDA output sequence

1. Read manifests and validate declared file checksums/row counts.
2. Build universe coverage from the security master and download status.
3. Build per-year observation and missingness summaries from validated data.
4. Audit corporate actions against raw and adjusted observations.
5. Calculate diagnostic returns in a temporary/per-security table only.
6. Generate figures and table fragments plus a machine-readable
   `report_statistics.json`.
7. Generate LaTeX prose from actual statistics; verify every number against
   `report_statistics.json`.
8. Compile and visually inspect every report page.

## Required final issue summary

The issue register must distinguish:

- `Critical`: identity collision or contamination that makes a security's data
  unusable without manual resolution.
- `High`: material historical-universe omission, broad adjustment error, or
  systematic volume convention issue.
- `Medium`: localized gap, ambiguous corporate-action event, or isolated
  discontinuity needing review.
- `Low`: documentation, formatting, or non-material metadata issue.
- `Informational`: observed but explained event, retained for audit.

Each item carries source, affected scope, evidence, status, and resolution.
