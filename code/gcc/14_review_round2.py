"""Addendum C (C3, C5): scales that down-weight the tail, heterogeneity, median bootstrap, tail frequency. Exploratory.
(1) heterogeneity (Q, I^2) and random-effects combination of the weekend change, on the mean and log scales;
(2) zero-gap shares by closure type and period, and a scale that keeps zeros: (E|z|_k / E|z|_N)^2;
(3) cluster-bootstrap (calendar weeks) standard errors for the median-based weekend change (plan 02 secondary);
(4) tail frequency: share of gaps with z^2 above the market's pre-switch 99th percentile, weekend vs weeknight, before vs after.
Output output/results_review_round2.json
"""
import pandas as pd, numpy as np, json, importlib.util, sys
from scipy import stats
spec = importlib.util.spec_from_file_location("est", "code/gcc/11_estimate_gcc.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
COMB = ["UAE", "KSA", "TASE_domestic", "TASE_dual"]

def meta(e, s):
    e, s = np.asarray(e), np.asarray(s); w = 1 / s ** 2
    fe = (w * e).sum() / w.sum(); Q = float((w * (e - fe) ** 2).sum()); k = len(e)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (s ** 2 + tau2); re = (wr * e).sum() / wr.sum()
    return dict(fixed=float(fe), fixed_se=float(1 / np.sqrt(w.sum())), Q=Q, Q_p=float(stats.chi2.sf(Q, k - 1)), I2=float(I2),
                tau2=float(tau2), random=float(re), random_se=float(1 / np.sqrt(wr.sum())))

def dW_from_cells(C, m, scale):
    """Delta log R_W from cell means of a statistic y; scale 'log' (y=log|z|), 'abs' (y=|z|) or 'sq' (y=z2)."""
    k = [f"{m}|1|W", f"{m}|1|N", f"{m}|0|W", f"{m}|0|N"]; sg = [1, -1, -1, 1]
    g = np.zeros(len(C.b))
    for key, s in zip(k, sg):
        i = C.ix[key]
        if scale == "log": g[i] += 2 * s
        elif scale == "abs": g[i] += 2 * s / C.b[i]
        else: g[i] += s / C.b[i]
    val = (2 * sum(s * C.b[C.ix[key]] for key, s in zip(k, sg))) if scale == "log" else \
          sum((2 if scale == "abs" else 1) * s * np.log(C.b[C.ix[key]]) for key, s in zip(k, sg))
    return float(val), C.se(g)

def main():
    g, t = est.load(); Cc = est.COLS
    prim = g[g.inwin == 1]
    uae = prim[prim.mk.isin(["DFM", "ADX"])].assign(mk="UAE"); ksa = prim[prim.mk == "KSA"]
    S = pd.concat([uae[Cc], ksa[Cc], t[Cc]], ignore_index=True)
    out = {}
    # (2) zero shares
    out["zero_share"] = (S.assign(zero=(S.z2 == 0).astype(float)).groupby(["mk", "post", "typ"]).zero.mean().round(3)
                         .rename(lambda x: "|".join(map(str, x)) if isinstance(x, tuple) else x).to_dict())
    out["zero_share"] = {"|".join(map(str, k)): v for k, v in S.assign(zero=(S.z2 == 0).astype(float)).groupby(["mk", "post", "typ"]).zero.mean().round(3).items()}
    # scales: mean (sq, winsorised), keep-zeros |z| (abs), log|z| on non-zero (log)
    S["az"] = np.sqrt(S.z2w)
    Cs = est.Cells(S, "z2w"); Ca = est.Cells(S, "az")
    Sl = S[S.z2w > 0].copy(); Sl["lz"] = 0.5 * np.log(Sl.z2w); Cl = est.Cells(Sl, "lz")
    res = {}
    for scale, C in (("mean", Cs), ("abs_keep_zeros", Ca), ("log_nonzero", Cl)):
        sc = {"mean": "sq", "abs_keep_zeros": "abs", "log_nonzero": "log"}[scale]
        d = {m: dW_from_cells(C, m, sc) for m in COMB}
        res[scale] = dict(per_market={m: dict(est=round(v[0], 3), se=round(v[1], 3)) for m, v in d.items()},
                          meta={k: round(v, 4) for k, v in meta([d[m][0] for m in COMB], [d[m][1] for m in COMB]).items()})
        # weeknight ratio on this scale
        wk = {}
        for m in COMB:
            for p in (0, 1):
                if C.has(f"{m}|{p}|Z", f"{m}|{p}|N"):
                    a, b = C.b[C.ix[f"{m}|{p}|Z"]], C.b[C.ix[f"{m}|{p}|N"]]
                    wk[f"{m}|{p}"] = round(float(np.exp(2 * (a - b)) if sc == "log" else ((a / b) ** 2 if sc == "abs" else a / b)), 3)
        res[scale]["R_Z"] = wk
    out["scales"] = res
    out["n_nonzero"] = {"|".join(map(str, k)): int(v) for k, v in Sl.groupby(["mk", "post", "typ"]).size().items()}
    # (4) tail frequency above the market's pre-switch 99th percentile of z2 (N and W pooled)
    tf = {}
    for m in COMB:
        d = S[S.mk == m].copy(); q = d[(d.post == 0) & d.typ.isin(["N", "W"])].z2w.quantile(0.99)
        d["ex"] = (d.z2w > q).astype(float)
        C = est.Cells(d[d.typ.isin(["N", "W"])], "ex")
        f = {k: round(float(C.b[C.ix[f"{m}|{k}"]]), 4) for k in ("0|N", "0|W", "1|N", "1|W")}
        l1, g1 = C.lr(f"{m}|1|W", f"{m}|1|N"); l0, g0 = C.lr(f"{m}|0|W", f"{m}|0|N")
        tf[m] = dict(freq=f, DeltaW_log=round(float(l1 - l0), 3), se=round(C.se(g1 - g0), 3))
    out["tail_frequency"] = tf
    # (3) cluster bootstrap over calendar weeks for the median-based weekend change
    rng = np.random.default_rng(11); B = int(sys.argv[1]) if len(sys.argv) > 1 else 299
    mb = {}
    for m in COMB:
        d = S[(S.mk == m) & (S.z2 > 0) & S.typ.isin(["N", "W"])][["week", "post", "typ", "az"]]
        def stat(x):
            md = x.groupby(["post", "typ"]).az.median()
            return 2 * (np.log(md[(1, "W")] / md[(1, "N")]) - np.log(md[(0, "W")] / md[(0, "N")]))
        obs = stat(d); weeks = d.week.unique(); groups = {w: idx for w, idx in d.groupby("week").groups.items()}
        bs = []
        for _ in range(B):
            ws = rng.choice(weeks, len(weeks)); idx = np.concatenate([groups[w] for w in ws])
            try: bs.append(stat(d.loc[idx]))
            except KeyError: pass
        bs = np.array(bs)
        mb[m] = dict(est=round(float(obs), 3), se=round(float(bs.std()), 3), ci=[round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])], B=len(bs))
    out["median_bootstrap"] = mb
    out["median_meta"] = {k: round(v, 4) for k, v in meta([mb[m]["est"] for m in COMB], [mb[m]["se"] for m in COMB]).items()}
    json.dump(out, open("output/results_review_round2.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
