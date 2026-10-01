# Addendum B to plan 02 — exploratory analyses decided after unsealing

Written 27 September 2026, after the primary results of plan 02 were computed
(output/results_gcc.json). Everything below is exploratory and will be labelled
so in the paper. It was prompted by an adversarial review of those results.

Primary results as unsealed (stock level, 22-day denominators, winsorised):
UAE ΔlogR_W = −0.07 (0.11); KSA −0.05 (0.22); combined with TASE −0.07 (0.08).
R_Z: UAE pre (Sunday nights) 1.18; UAE post (US-holiday nights) 0.70; KSA pre 0.72,
post 0.74; TASE domestic 1.08, dual-listed 0.53. H1 not supported in any market;
H2 supported in KSA (both regimes), UAE post, TASE dual-listed; contradicted in UAE pre.

B1 Bound. Per market, the weeknight per-session effect δ = 1 − R_Z and the weekend
per-session effect implied by the switch, δ_W = R_W,pre (1 − e^Δ)/ΔU, with delta-method
SEs; a 95% upper bound on δ_W from the combined estimate.
B2 US holidays. Drop 24 Dec–2 Jan and Good Friday; report Monday US holidays alone.
B3 UAE Sunday nights. (i) Sunday vs other weekdays: volume, share of zero intraday
returns, intraday variance; (ii) reversal: gap on the preceding session's open-to-close
return, by night; (iii) denominator built without Sunday sessions; (iv) Sunday-night z²
by tercile of Sunday volume relative to the firm's median.
B4 Denominators: 66-session window.
B5 Saudi price limits: drop gaps with |g| ≥ 9.5%.
B6 Ramadan: drop Ramadan months.
B7 TASE confound: the post-switch Israeli weekend closure contains Sunday, a working
day; stated in the text, no test.
