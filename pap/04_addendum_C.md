# Addendum C — further robustness and scale checks (exploratory)

Decided after the primary results of plan 02 and Addendum B were known. Exploratory throughout.
Code: code/gcc/13_review_checks.py, 14_review_round2.py, 15_abs_scale.py, 16_median_bootstrap.py, 17_close_to_close.py.

C1 Saudi post-period split at the QFI opening (15 June 2015).
C2 Exclusion of the weeks of 7 Oct–31 Dec 2023, 13–30 Apr 2024 and 13–30 Jun 2025.
C3 Scales that down-weight the tail: (E|z|_k/E|z|_N)^2 on all gaps; exp(2(E log|z|_k − E log|z|_N)) on non-zero gaps;
   median-based ratios (plan 02 secondary) with a cluster bootstrap over calendar weeks (999 draws); heterogeneity (Q, I², random effects).
C4 First night of the week: ordinary weeknights by the weekday of the following session.
C5 Share of the sum of z² contributed by the largest 1% of share-gaps; frequency of gaps above the pre-switch 99th percentile.
C6 Close-to-close weekend measure: log(C_t/C_{t−1}) normalised by the mean squared close-to-close return of the 22 previous sessions,
   on the mean and |c| scales, to count information priced during the first session after a closure.
