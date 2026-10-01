# Pre-analysis plan

**Working title:** Time or trading? Evidence on weekend return variance from the Tel Aviv Stock Exchange's move to a Monday–Friday week.

**Written:** 2026-09-23, before any estimation on outcome variables. Descriptive work done so far: tabulating trading days per weekday, and checking official TASE prices against Yahoo Finance on three dates. No variances, regressions or mean returns have been computed.

## 1. Setting
Until 1 January 2026 the Tel Aviv Stock Exchange (TASE) traded Sunday to Thursday. Since 5 January 2026 it trades Monday to Friday. The Friday session closes with an auction at 13:45 Israel time, before the NYSE opens. Monday–Thursday hours did not change: the closing auction ends at about 17:25, roughly one hour after the NYSE open. Sunday sessions closed at about 15:50 from 8 March 2020.

Consider each non-trading interval ("gap"), from the TASE close on the previous session to the TASE open on the current session.

- The **weekend gap before the switch** (Thursday close to Sunday open) lasts about 64.6 hours. It contains about 1.86 NYSE regular sessions: the Thursday session from about 10:25 New York time, plus all of Friday.
- The **weekend gap after the switch** (Friday close to Monday open) lasts about 68.2 hours. It contains exactly one NYSE session, Friday.
- **Ordinary weeknight gaps** last about 16.6 hours and contain about 0.86 NYSE sessions.
- The **Sunday-night gap before the switch** (Sunday close to Monday open) lasts about 18.2 hours and contains no NYSE session.

The switch therefore holds the calendar length of the weekend roughly fixed (+6%) while nearly halving the foreign trading time inside it. The pre-switch Sunday night provides a gap of weeknight length with no foreign trading.

## 2. Hypotheses
Let $g_t = \log(O_t / C_{t-1})$ be the gap return and $d_t = \log(C_t/O_t)$ the intraday return of the TA-125 index.

- **H-cal (calendar time).** Gap variance is proportional to elapsed calendar hours $H_t$.
- **H-info (foreign information arrival).** Gap variance is proportional to the foreign trading time $U_t$ inside the gap, measured in NYSE sessions.

Predictions for the relative weekend variance, $R = \mathrm{Var}(g \mid \text{weekend}) / \mathrm{Var}(g \mid \text{ordinary weeknight})$, before and after the switch:
- under H-cal, $R$ is about 3.9 before and about 4.1 after;
- under H-info, $R$ is about 2.2 before and about 1.2 after.

## 3. Data
- Official TASE end-of-day data (base, open, close, high, low) for TA-125 (primary) and TA-35 (robustness), 2005-01-02 to 2026-09-23, from the TASE website's historical-data service.
- S&P 500 from Yahoo Finance (^GSPC). Used only to date NYSE sessions and to compute NYSE close-to-close returns. The Yahoo record for TASE is **not** used for estimation, because it omits Sunday sessions (see §6).
- NYSE session times are 9:30–16:00 New York time. Half-days are counted as full sessions (a known small error). Clock times use zoneinfo (Asia/Jerusalem, America/New_York), so the offsets are correct when the two countries' daylight-saving changes fall on different dates.

## 4. Sample
- **Primary sample:** 2020-03-08 to 2026-09-23, the period with constant Sunday hours. Pre = before 2026-01-05; post = from 2026-01-05.
- **Gap types:**
  - W-pre: first session after a Friday–Saturday closure, Thursday close to Sunday open.
  - W-post: Friday close to Monday open.
  - N: ordinary weeknight, next calendar day, the previous session not a Sunday before the switch, and not a Friday after it.
  - S-pre: Sunday-night gap, Sunday close to Monday open.
  - H: any other gap, including holidays. H gaps are excluded from the main tests and kept for the structural model.
- **Robustness samples:**
  - (a) excluding 2020;
  - (b) excluding 2023-10-07 to 2023-12-31 (outbreak of war);
  - (c) excluding June 2025 (Israel–Iran war);
  - (d) the long pre sample, 2005-01-02 to 2025-12-31, with approximate hours.

## 5. Specifications (primary first)

**P1. Variance ratio test.** Normalise by the local volatility of the index: $z_t = g_t / \hat\sigma_t$, where $\hat\sigma_t^2$ is the mean of $d_s^2$ over the 22 sessions before $t$. Estimate $R_{pre}$ and $R_{post}$ as ratios of mean $z^2$ for W versus N gaps. Test $H_0$: $R_{post} = R_{pre}$, the change predicted to be zero under H-cal.
- Inference: bootstrap that resamples weeks (stationary block bootstrap, mean block length 4 weeks, 9,999 draws), plus a regression-based test.
  - Regression: $z_t^2 = a + b\,W_t + c\,\mathrm{Post}_t + \delta\,W_t\times \mathrm{Post}_t + e\,S_t + u_t$ on N, W and S-pre gaps, with HAC standard errors (Newey–West, 10 lags).
  - The test is on $\delta$. H-info predicts $\delta < 0$; the magnitude implied by the prediction ratios is roughly a halving of $R$.
- Report the result in levels as well, together with a Brown–Forsythe test.

**P2. Structural decomposition.** Pool all gaps of type N, W and S, together with post Thursday nights (a type N gap). Estimate
$$E[z_t^2] = \kappa + \lambda_H H_t + \lambda_U U_t$$
by OLS with HAC standard errors. Identification of $\lambda_H$ separately from $\lambda_U$ comes from the S-pre gaps and the switch.
- Report the implied share of weekend gap variance attributable to calendar hours.
- Pre-registered expectation: $\lambda_U > 0$ and $\lambda_H$ small. Any outcome is reported.

**P3. Loading on US returns.** Regress $g_t = \alpha + \sum_k \beta_k X_{k,t} + \varepsilon_t$ on the NYSE close-to-close returns of the sessions inside the gap: for W-pre the US Thursday and Friday returns, for W-post the US Friday return. The timing prediction is that the pre-switch weekend gap loads on US Thursday and Friday, and the post-switch gap only on US Friday.
- Then compare the variance of the residuals of W versus N gaps with the P1 procedure. This asks whether any excess weekend variance remains after projecting out US information.

**P4. Means (secondary, known low power).** Mean of $g_t$ and of the close-to-close return for the first session after the weekend, pre versus post, reported with standard errors. No claim will rest on this test.

**P5. Data-vendor exhibit (descriptive).** Compare Yahoo Finance ^TA125.TA with the official series:
- count the sessions that Yahoo omits;
- run a standard day-of-week regression on each source, 2014–2025, and report the difference.

## 6. Robustness (fixed list)
1. TA-35 in place of TA-125.
2. Outcome $g_t^2$ without normalisation.
3. $\hat\sigma_t$ computed from 66 sessions instead of 22.
4. Samples (a)–(d) above.
5. Winsorising $z_t$ at the 1st and 99th percentiles.
6. Excluding gaps adjacent to Israeli holidays.
7. Post-period placebo: the switch date moved to 2025-01-05 on pre data only.

## 7. Multiple testing and reporting
- The headline test is P1 ($\delta$); everything else is secondary.
- Every result above will be reported whether or not it is significant. Any additional analysis will be labelled exploratory.
- A null result on P1, given the power of about 37 post-switch weekends, would mean the post-switch sample cannot yet separate the hypotheses. It will be reported with a minimum detectable effect.

---
## Addendum A (2026-09-23; written after the index-level P1–P5 results, before any stock-level data were analysed)

**Why.** The index-level P1 test has low power: 34 weekends after the switch give a bootstrap SE of about 0.45 on the log-ratio. The index-level results also showed that the Sunday-night gap before the switch, which contains no NYSE cash session, had variance similar to ordinary weeknights. This raised the point, not stated in the original plan, that US equity-index futures (CME Globex, Sunday 18:00 to Friday 17:00 New York time, with a daily halt from 17:00 to 18:00) and Asian Monday sessions trade inside that gap. The additions below are therefore labelled **post-registration**.

**A1. Stock-level triple difference (the new primary stock-level test).**
- Universe: current TA-125 constituents, with daily TASE data from 2020-01-01. Survivorship is noted as a limitation.
- Groups:
  - DL: TASE's own "Dual-Listed Company" flag, 16 firms. These shares trade abroad, mostly on NASDAQ/NYSE, during TASE closures.
  - Non-DL: all other constituents.
- Outcome: $z^2_{it} = g_{it}^2/\hat\sigma^2_{it}$, where $g_{it} = \log(\text{open}_{it}/\text{base}_{it})$ and $\hat\sigma^2_{it}$ is the mean of the stock's own intraday $d^2$ over the 22 previous sessions. Stock-gaps with no trade (IfTraded = 0), or with |g| > 25%, are dropped.
- Statistic: $\Delta = [\log R_{DL,post} - \log R_{DL,pre}] - [\log R_{nonDL,post} - \log R_{nonDL,pre}]$, where $R$ is the pooled W/N ratio of mean $z^2$ within a group and period.
  - Under H-info, $\Delta < 0$: the shares of DL firms themselves trade for two sessions abroad during the pre-switch weekend and for one after it.
  - Under H-cal, $\Delta = 0$.
- Inference: stationary block bootstrap that resamples calendar weeks jointly across all stocks (mean block 4 weeks, 4,999 draws).
- Also reported: $R$ by group and period, and pooled regression of $z^2$ on $W$, Post, $W\times$Post, DL and their interactions, with standard errors clustered by week.

**A2. Alternative foreign-trading clock (exploratory).** $U^{fut}$ = hours of CME equity-index futures trading inside the gap. Re-estimate P2 with $U^{fut}$ in place of $U$.

**A3. Robust scale (exploratory).** Rerun P1 with $\log|z|$ as the outcome, as a heavy-tail-robust scale comparison.
