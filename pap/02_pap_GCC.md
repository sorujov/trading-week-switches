# Pre-analysis plan 02 — the Gulf trading-week switches (UAE 2022, Saudi Arabia 2013)

Written 27 September 2026, before any gap-return variance for a Gulf market
has been computed. At the time of writing the only Gulf data examined are
the outputs of LSEG job 01 (session dates by weekday and the share of index
opens equal to the previous close), which are data-quality checks and
contain no gap-variance statistic. The SHA-256 of this file is recorded in
`pap/HASHES.txt` and in the project log. It is pre-specified and
hash-timestamped; it is not registered with a third party, and the paper
will say so.

Plan 01 (`01_pap_TASE.md`, with Addendum A) governs the Tel Aviv switch and
is not changed by this document, except that §6 below adds the pooled test.

---

## 1. Question

Whether the variance of returns over a market closure accrues in calendar
time or in trading time abroad. A trading-week switch keeps the weekend
closure at roughly the same length but changes how many US cash sessions it
contains; the old or new calendar also contains weeknight closures with no
US session.

## 2. Events and regimes

| market | last day old week | first day new week | old week | new week |
|---|---|---|---|---|
| UAE: DFM and ADX | Thu 30 Dec 2021 | Mon 3 Jan 2022 | Sun–Thu | Mon–Fri |
| Saudi Arabia: Tadawul | Wed 26 Jun 2013 | Sun 30 Jun 2013 | Sat–Wed | Sun–Thu |

Session hours and dates are taken from exchange notices and verified
against the session dates present in the data (step 1). If a verified date
differs from this table, the verified date is used and the difference is
reported; nothing else in the plan changes.

**Windows.**
- UAE: primary window 2 Jan 2018 – 25 Sep 2026 (four years before, all
  available after). Robustness: drop 23 Feb – 30 Jun 2020 (Covid crash);
  symmetric window of 3 years 9 months either side.
- Saudi Arabia: primary window ±4 years (1 Jul 2009 – 30 Jun 2017).
  Robustness: ±2 years (1 Jul 2011 – 30 Jun 2015), which ends before the
  opening to qualified foreign investors in June 2015; ±5 years.

## 3. Units and data

**Primary unit: constituent stocks.** For each market (DFM, ADX, Tadawul),
the 30 largest firms by market capitalisation on the last day of the
primary window whose price history covers the whole primary window, taken
from LSEG. This list is fixed by rule before any gap statistic is computed
and is saved to `data/raw/lseg/`. The survivorship this implies is stated
in the paper; it is not expected to bear on a ratio of closure variances.

**Secondary unit: the index** (DFM General Index, FTSE ADX General Index,
TASI). If, in any year of a window, more than 5% of an index's opens equal
the previous close to six digits, index results for that market are
reported only in an appendix.

**Observation.** A gap g = log(open_t) − log(close_{t−1}) on consecutive
sessions of the same security. Excluded: a gap where either price is
missing, where volume on day t is zero, or where the open equals the
previous close exactly and volume on day t is below the stock's 5th
percentile (a no-trade open).

**Local scale.** σ̂²_t is the mean of squared open-to-close log returns over
the 22 sessions before t (at least 15 required). z²_t = g²_t / σ̂²_t,
winsorised at the 99.9th percentile of the market's pooled z².

## 4. Closure types

Each closure (close of t−1 to open of t) is described by
- H: calendar hours;
- U: the number of NYSE cash sessions inside it, counting a session
  fraction by the share of its hours that fall inside the closure, NYSE
  half days counted at their length;
- W: 1 if the closure contains at least one day that is a weekend day of
  the market's calendar in force.

Types: **N** ordinary weeknight (W = 0, U = 1); **Z** weeknight with no US
session (W = 0, U = 0: the UAE Sunday night before the switch; the Saudi
Saturday and Sunday nights before, the Sunday night after); **W** weekend
closure with no holiday; **H** any closure containing an exchange holiday
(excluded from the primary tests).

Implied values (to be recomputed from verified hours):

| | UAE before | UAE after | Saudi before | Saudi after |
|---|---|---|---|---|
| weekend U | 2 (Thu, Fri) | 1 (Fri) | 3 (Wed, Thu, Fri) | 2 (Thu, Fri) |
| Z closures | Sun night | none | Sat, Sun nights | Sun night |

## 5. Hypotheses and tests (per market)

Let R_k = mean z² over type-k closures divided by mean z² over type-N
closures, within market × regime.

**H1 (weekend).** Under trading time the weekend ratio falls when the
weekend loses a US session: Δ log R_W = log R_W(after) − log R_W(before)
< 0. Under calendar time Δ log R_W = log(H_W(after)/H_W(before)), close to
zero. One-sided test at 5%.

**H2 (zero-US weeknight).** Under trading time R_Z < 1; under calendar
time R_Z ≈ H_Z / H_N ≈ 1. One-sided test of log R_Z < 0 in every regime
that has Z closures.

**Inference.** Stock level: z² regressed on type × regime dummies with
standard errors clustered two ways (calendar week, firm); ratios and their
log-differences by the delta method. Index level and as a check: stationary
block bootstrap over calendar weeks, mean block length 4 weeks, 1,999
draws. Firm-level randomisation inference is not available here (no
dual-listed/domestic split), so none is claimed.

**Secondary (reported, not tested for significance).** The same ratios
computed on log|z| (median-based), which is robust to the heavy tails of
gap returns; and close-to-close weekday variances.

## 6. Pooled test across the three switches

The model E[z² | closure] = a_m + b·U + c·W, with a market-specific
intercept a_m and common b and c, is estimated on the TASE, UAE and Saudi
stock panels pooled (TASE: domestic and dual-listed separately, since the
TASE plan already treats them separately). Primary pooled tests:
1. b > 0 (one-sided);
2. the calendar-time restriction that every ratio equals its hours ratio,
   R_k = H_k/H_N for all k, as a joint Wald test (two-sided).

The switch-by-switch estimates of Δ log R_W are combined by inverse-variance
weighting and shown together in one figure. No switch is dropped from the
pooled analysis because of its result.

## 7. Exploratory (labelled as such in the paper)

- A second clock: ICE Brent futures trading hours inside each closure in
  place of U.
- Heterogeneity by firm size, foreign ownership limit or sector (banks,
  real estate, petrochemicals).
- The Friday session in the UAE after 2022, compared with Monday–Thursday.

## 8. What would count against the claim

Δ log R_W ≥ 0 in both Gulf switches, or R_Z not below 1 where Z closures
exist, or b not positive in the pool. Any of these would be reported in the
abstract.
