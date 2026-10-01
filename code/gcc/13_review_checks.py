"""Addendum C (C1–C5): further robustness checks. Exploratory.
(a) Saudi post-period split at the QFI opening (15 June 2015); (b) exclusion of geopolitical shock windows;
(c) a tail-robust version of the main ratios based on log|z|; (d) first-night-of-the-week check.
Output output/results_review_checks.json
"""
import pandas as pd, numpy as np, json, importlib.util
from scipy import stats
spec = importlib.util.spec_from_file_location("est", "code/gcc/11_estimate_gcc.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)

def summ(r):
    o = {k: dict(R=round(v["R"], 3), se=round(v["se"], 3)) for k, v in r["R"].items()}
    if "DeltaW" in r: o["DeltaW"] = dict(est=round(r["DeltaW"]["est"], 3), se=round(r["DeltaW"]["se"], 3))
    return o

def main():
    g, t = est.load(); C = est.COLS
    prim = g[g.inwin == 1]
    uae = prim[prim.mk.isin(["DFM", "ADX"])].assign(mk="UAE"); ksa = prim[prim.mk == "KSA"]
    out = {}
    # (a) Saudi: post period before and after 15 June 2015, each against the same pre period
    for lab, lo, hi in (("post_2013_2015", "2013-06-30", "2015-06-14"), ("post_2015_2017", "2015-06-15", "2017-06-30")):
        d = ksa[(ksa.post == 0) | ksa.date.between(lo, hi)]
        out[f"KSA_{lab}"] = summ(est.estimate(d[C], ["KSA"])["KSA"])
    # (b) geopolitical windows excluded (Israel 7 Oct - 31 Dec 2023; Iran-Israel strikes 13-30 Apr 2024 and 13-30 Jun 2025)
    def drop(df):
        m = df.date.between("2023-10-07", "2023-12-31") | df.date.between("2024-04-13", "2024-04-30") | df.date.between("2025-06-13", "2025-06-30")
        return df[~m]
    stack = pd.concat([drop(uae)[C], ksa[C], drop(t)[C]], ignore_index=True)
    comb = ["UAE", "KSA", "TASE_domestic", "TASE_dual"]
    r = est.estimate(stack, comb, combine=comb); r.pop("_C")
    out["geo_excluded"] = {k: summ(r[k]) for k in comb}; out["geo_excluded"]["combined"] = {k: round(r["DeltaW_combined"][k], 3) for k in ("est", "se")}
    # (c) log|z| version: scale ratio exp(2 (E log|z|_k - E log|z|_N)), zero gaps excluded
    def logz(df):
        d = df[df.z2w > 0].copy(); d["lz"] = 0.5 * np.log(d.z2w); return d
    L = pd.concat([logz(uae)[C + []].assign(lz=logz(uae).lz.values), logz(ksa)[C].assign(lz=logz(ksa).lz.values),
                   logz(t)[C].assign(lz=logz(t).lz.values)], ignore_index=True)
    Cl = est.Cells(L, "lz")
    lzres = {}
    for m in comb:
        o = {}
        for p in (0, 1):
            for tt in ("W", "Z"):
                k, n = f"{m}|{p}|{tt}", f"{m}|{p}|N"
                if Cl.has(k, n):
                    gv = np.zeros(len(Cl.b)); gv[Cl.ix[k]] = 2; gv[Cl.ix[n]] = -2
                    d = float(gv @ Cl.b); o[f"{tt}_post{p}"] = dict(R=round(float(np.exp(d)), 3), se_log=round(Cl.se(gv), 3))
        if Cl.has(*(f"{m}|{p}|{x}" for p in (0, 1) for x in ("W", "N"))):
            gv = np.zeros(len(Cl.b))
            for key, s in ((f"{m}|1|W", 2), (f"{m}|1|N", -2), (f"{m}|0|W", -2), (f"{m}|0|N", 2)): gv[Cl.ix[key]] += s
            o["DeltaW"] = dict(est=round(float(gv @ Cl.b), 3), se=round(Cl.se(gv), 3))
        lzres[m] = o
    # combined weekend change on the log|z| scale, joint covariance; weeknight delta on that scale and implied prediction
    grads, ests = [], []
    for m in comb:
        gv = np.zeros(len(Cl.b))
        for key, sg in ((f"{m}|1|W", 2), (f"{m}|1|N", -2), (f"{m}|0|W", -2), (f"{m}|0|N", 2)): gv[Cl.ix[key]] += sg
        grads.append(gv); ests.append(float(gv @ Cl.b))
    G = np.array(grads); S = G @ Cl.V @ G.T; w = (1 / np.diag(S)) / (1 / np.diag(S)).sum()
    lzres["combined"] = dict(est=round(float(w @ np.array(ests)), 3), se=round(float(np.sqrt(w @ S @ w)), 3))
    zkey = {"UAE": "Z_post1", "KSA": "Z_post0", "TASE_domestic": "Z_post0", "TASE_dual": "Z_post0"}
    dU = {"UAE": 0.966, "KSA": 1.021, "TASE_domestic": 0.858, "TASE_dual": 0.860}
    for m in comb:
        dz = 1 - lzres[m][zkey[m]]["R"]; Rw0 = lzres[m]["W_post0"]["R"]
        lzres[m]["delta_weeknight"] = round(dz, 3); lzres[m]["pred_DeltaW"] = round(float(np.log(max(1 - dz * dU[m] / Rw0, 1e-6))), 3)
    out["logabs"] = lzres
    # median-based ratios (pre-specified secondary), non-zero gaps
    med = {}
    for m, df in (("UAE", uae), ("KSA", ksa), ("TASE_domestic", t[t.mk == "TASE_domestic"]), ("TASE_dual", t[t.mk == "TASE_dual"])):
        d = df[df.z2 > 0].assign(az=lambda x: np.sqrt(x.z2)); mm = d.groupby(["post", "typ"]).az.median()
        med[m] = {f"{tt}_post{p}": round(float((mm[(p, tt)] / mm[(p, "N")]) ** 2), 3) for p in (0, 1) for tt in ("W", "Z") if (p, tt) in mm.index}
    out["median_R_nonzero"] = med
    # how much of the mean weekend z2 comes from the largest 1% of weekend share-gaps
    tail = {}
    for m, df in (("UAE", uae), ("KSA", ksa), ("TASE_domestic", t[t.mk == "TASE_domestic"]), ("TASE_dual", t[t.mk == "TASE_dual"])):
        for p in (0, 1):
            for tt in ("W", "N"):
                x = np.sort(df[(df.post == p) & (df.typ == tt)].z2w.values)[::-1]
                k = max(1, int(round(0.01 * len(x))))
                tail[f"{m}|{p}|{tt}"] = round(float(x[:k].sum() / x.sum()), 3)
    out["top1pct_share_of_sum"] = tail
    # (d) first night of the week: ordinary weeknights by weekday of the closing session, relative to all N nights
    fn = {}
    for m, df in (("UAE", uae), ("KSA", ksa)):
        for p in (0, 1):
            d = df[(df.post == p) & (df.typ == "N")].copy()
            d["typ"] = pd.to_datetime(d.date).dt.day_name().str[:3]   # weekday of the session that follows the night
            d2 = pd.concat([d, df[(df.post == p) & (df.typ == "N")]])  # pooled N as the reference
            ref = df[(df.post == p) & (df.typ == "N")].z2w.mean()
            fn[f"{m}_post{p}"] = {k: dict(R=round(v.z2w.mean() / ref, 3), n=int(len(v))) for k, v in d.groupby("typ")}
    out["weeknights_by_following_session"] = fn
    json.dump(out, open("output/results_review_checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
