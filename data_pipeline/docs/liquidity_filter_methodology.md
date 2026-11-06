# Liquidity-filter methodology

The fixed screen uses the final 20 XNYS sessions ending
2022-12-30. A row is valid only for a finite positive nominal
close and finite nonnegative reported share volume. Missing values and absent
sessions are not substituted or filled. A valid zero-volume row contributes
zero dollar volume.

Yahoo's nominal Close is split-back-adjusted while Volume is treated as
reported contemporaneous share volume. Each window Close is restored to the
observation-date share basis by multiplying future split factors dated after
that observation through 2025-12-31.
Daily dollar volume is then compatible Close times reported Volume. This uses
future corporate-action metadata only to interpret historical vendor
adjustments, never future prices, volumes, or returns.

The inclusive threshold is USD 100,000,000. Of
3,239 acquired securities, 488 selected into the
fixed 2016-2025 panel containing 1,187,986 validated rows. All raw
and previously validated files remain unchanged.
