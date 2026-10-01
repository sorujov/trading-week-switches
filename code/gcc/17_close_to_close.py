"""Close-to-close check (editor's request): closure-plus-first-session returns c_t = log(C_t / C_{t-1}),
normalised by the mean of c^2 over the previous 22 sessions, so that information priced during the first session
after a closure is counted with the closure. Ratios by closure type on the mean scale (z^2) and on the |z| scale.
Output output/results_close_to_close.json"""
import pandas as pd, numpy as np, json, glob, os, importlib.util
spec = importlib.util.spec_from_file_location("est", "code/gcc/11_estimate_gcc.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
COMB = ["UAE", "KSA", "TASE_domestic", "TASE_dual"]
def main():
    g, t = est.load()
    g = g[g.inwin == 1]
    # Gulf: rebuild close-to-close from the raw files for the share-gaps already selected
    rows = []
    for (market, firm), d in g.groupby(["market", "firm"]):
        s = pd.read_csv(f"data/raw/lseg/job02/stocks/{market}__{firm}.csv", index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
        s.index = s.index.normalize(); s = s[~s.index.duplicated()].sort_index()
        c = np.log(s.TRDPRC_1 / s.TRDPRC_1.shift(1)); c[~np.isfinite(c)] = np.nan
        sc = (c ** 2).shift(1).rolling(22, min_periods=15).mean()
        x = pd.DataFrame({"date": s.index, "cc": c.values, "scc": sc.values})
        rows.append(d.merge(x, on="date", how="left"))
    G = pd.concat(rows)
    # TASE: close-to-close from base price (previous close adjusted) to close
    tt = []
    for f in glob.glob("data/raw/tase/stocks/*.csv"):
        sec = os.path.basename(f)[:-4]
        s = pd.read_csv(f, parse_dates=["date"]).sort_values("date").drop_duplicates("date")
        s = s[(s.base > 0) & (s.close > 0)].copy()
        s["cc"] = np.log(s.close / s.base); s["scc"] = (s.cc ** 2).shift(1).rolling(22, min_periods=15).mean()
        tt.append(s[["date", "cc", "scc"]].assign(firm="TA-" + sec))
    T = t.merge(pd.concat(tt), on=["firm", "date"], how="left")
    G["mk"] = np.where(G.market == "TASI", "KSA", "UAE")
    S = pd.concat([G[est.COLS + ["cc", "scc"]], T[est.COLS + ["cc", "scc"]]], ignore_index=True).dropna(subset=["cc", "scc"])
    S = S[S.scc > 0]
    S["c2"] = S.cc ** 2 / S.scc
    q = S.groupby("mk").c2.transform(lambda x: x.quantile(0.999)); S["c2w"] = np.minimum(S.c2, q); S["ac"] = np.sqrt(S.c2w)
    out = {}
    for name, y, sq in (("mean_scale", "c2w", True), ("abs_scale", "ac", False)):
        C = est.Cells(S, y); res = {}
        for m in COMB:
            o = {}
            def R(p, k):
                a, b = C.b[C.ix[f"{m}|{p}|{k}"]], C.b[C.ix[f"{m}|{p}|N"]]
                return float(a / b if sq else (a / b) ** 2)
            for p in (0, 1):
                for k in ("W", "Z"):
                    if f"{m}|{p}|{k}" in C.ix: o[f"{k}_post{p}"] = round(R(p, k), 3)
            gv = np.zeros(len(C.b)); f_ = 1 if sq else 2
            for key, sgn in ((f"{m}|1|W", 1), (f"{m}|1|N", -1), (f"{m}|0|W", -1), (f"{m}|0|N", 1)):
                gv[C.ix[key]] += f_ * sgn / C.b[C.ix[key]]
            est_ = float(sum(f_ * sgn * np.log(C.b[C.ix[key]]) for key, sgn in ((f"{m}|1|W", 1), (f"{m}|1|N", -1), (f"{m}|0|W", -1), (f"{m}|0|N", 1))))
            o["DeltaW"] = dict(est=round(est_, 3), se=round(C.se(gv), 3)); o["_g"] = gv
            res[m] = o
        G_ = np.array([res[m].pop("_g") for m in COMB]); Sg = G_ @ C.V @ G_.T
        e = np.array([res[m]["DeltaW"]["est"] for m in COMB]); w = (1 / np.diag(Sg)) / (1 / np.diag(Sg)).sum()
        res["combined"] = dict(est=round(float(w @ e), 3), se=round(float(np.sqrt(w @ Sg @ w)), 3))
        out[name] = res
    out["n"] = int(len(S))
    # Saudi post period split at the QFI opening (15 June 2015), each against the same pre period
    K = S[S.mk == "KSA"]
    for lab, lo, hi in (("KSA_post_2013_2015", "2013-06-30", "2015-06-14"), ("KSA_post_2015_2017", "2015-06-15", "2017-06-30")):
        d = K[(K.post == 0) | K.date.between(lo, hi)]; o = {}
        for name, y, sq in (("mean_scale", "c2w", True), ("abs_scale", "ac", False)):
            C = est.Cells(d, y); f_ = 1 if sq else 2; gv = np.zeros(len(C.b)); e = 0.0
            for key, sgn in (("KSA|1|W", 1), ("KSA|1|N", -1), ("KSA|0|W", -1), ("KSA|0|N", 1)):
                gv[C.ix[key]] += f_ * sgn / C.b[C.ix[key]]; e += f_ * sgn * np.log(C.b[C.ix[key]])
            o[name] = dict(est=round(float(e), 3), se=round(C.se(gv), 3))
        out[lab] = o
    json.dump(out, open("output/results_close_to_close.json", "w"), indent=1); print(json.dumps(out, indent=1))
if __name__ == "__main__":
    main()
