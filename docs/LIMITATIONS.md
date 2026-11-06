# Evidence and limitations

- 2023 is exposed development validation. Repeated inspection and selection
  invalidate an independent-test interpretation, even with past-only fitting.
- The reference-date universe is fixed retrospectively. Missing/delisted
  securities, ticker continuity and non-random acquisition failures create
  survivorship and selection concerns. Training is not point-in-time unbiased.
- Results contain a large passive core. Reported returns and Sharpe are not
  attributable to the active sleeve or to ML alone.
- Training maximum drawdown is approximately 34%; the smaller 2023 drawdown is
  not a forward risk bound. Active target caps can be breached after gaps.
- Corporate actions, share-volume units and vendor repairs remain imperfect.
  Adjusted-close execution, fractional shares and a fixed proportional fee
  omit real execution capacity, market impact and detailed slippage.
- Completed-trade labels exclude open segments, creating right-censoring.
- Snapshot reconstruction is distinct from a fresh vendor download. Raw and
  intermediate data are not distributed in the public source package.
- The fixed-model 2024–2025 test has now been evaluated and is documented in
  TEST_REPORT.md. Future revisions informed by these results cannot treat the
  interval as untouched. This does not establish readiness for live trading.
