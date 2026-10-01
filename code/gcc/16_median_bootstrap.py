"""Cluster bootstrap (calendar weeks) for the median-based weekend change, in chunks so that each run fits a short job.
Usage: python code/gcc/16_median_bootstrap.py <draws> <seed>   (appends draws to output/median_boot_draws.csv)
       python code/gcc/16_median_bootstrap.py summarise          (writes output/results_median_bootstrap.json)"""
import pandas as pd, numpy as np, json, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("est", "code/gcc/11_estimate_gcc.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
spec2 = importlib.util.spec_from_file_location("r2", "code/gcc/14_review_round2.py"); r2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(r2)
COMB = ["UAE", "KSA", "TASE_domestic", "TASE_dual"]; F = "output/median_boot_draws.csv"
def data():
    g, t = est.load(); C = est.COLS; prim = g[g.inwin == 1]
    uae = prim[prim.mk.isin(["DFM", "ADX"])].assign(mk="UAE"); ksa = prim[prim.mk == "KSA"]
    S = pd.concat([uae[C], ksa[C], t[C]], ignore_index=True); S["az"] = np.sqrt(S.z2w)
    return S[(S.z2 > 0) & S.typ.isin(["N", "W"])][["mk", "week", "post", "typ", "az"]]
def stat(x):
    md = x.groupby(["post", "typ"]).az.median()
    return 2 * (np.log(md[(1, "W")] / md[(1, "N")]) - np.log(md[(0, "W")] / md[(0, "N")]))
if sys.argv[1] == "summarise":
    D = pd.read_csv(F); S = data(); out = {}
    for m in COMB:
        b = D[m].dropna().values; out[m] = dict(est=round(float(stat(S[S.mk == m])), 3), se=round(float(b.std()), 3),
                                               ci=[round(float(v), 3) for v in np.percentile(b, [2.5, 97.5])], B=int(len(b)))
    out["meta"] = {k: round(v, 4) for k, v in r2.meta([out[m]["est"] for m in COMB], [out[m]["se"] for m in COMB]).items()}
    json.dump(out, open("output/results_median_bootstrap.json", "w"), indent=1); print(out); sys.exit()
B, seed = int(sys.argv[1]), int(sys.argv[2]); rng = np.random.default_rng(seed); S = data(); rows = {m: [] for m in COMB}
for m in COMB:
    d = S[S.mk == m].reset_index(drop=True); weeks = d.week.unique(); gr = {w: i.values for w, i in d.groupby("week").groups.items()}
    for _ in range(B):
        ws = rng.choice(weeks, len(weeks)); idx = np.concatenate([gr[w] for w in ws])
        try: rows[m].append(stat(d.iloc[idx]))
        except KeyError: rows[m].append(np.nan)
pd.DataFrame(rows).to_csv(F, mode="a", header=not os.path.exists(F), index=False); print("appended", B)
