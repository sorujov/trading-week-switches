"""Addendum C, second round: ratios on the |z| scale keeping zero gaps, R_k = (E|z|_k / E|z|_N)^2, with the
weekend change implied by each market's weeknight contrast on the same scale. Output output/results_abs_scale.json"""
import importlib.util, numpy as np, pandas as pd, json
spec = importlib.util.spec_from_file_location("est", "code/gcc/11_estimate_gcc.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
g, t = est.load(); C = est.COLS; prim = g[g.inwin == 1]
uae = prim[prim.mk.isin(["DFM", "ADX"])].assign(mk="UAE"); ksa = prim[prim.mk == "KSA"]
S = pd.concat([uae[C], ksa[C], t[C]], ignore_index=True); S["az"] = np.sqrt(S.z2w)
Ca = est.Cells(S, "az"); out = {}
dU = {"UAE": 0.966, "KSA": 1.021, "TASE_domestic": 0.858, "TASE_dual": 0.860}   # US sessions lost by the weekend (Table 1)
zk = {"UAE": "1", "KSA": "0", "TASE_domestic": "0", "TASE_dual": "0"}
E = np.eye(len(Ca.b))
for m in dU:
    R = lambda p, k: float((Ca.b[Ca.ix[f"{m}|{p}|{k}"]] / Ca.b[Ca.ix[f"{m}|{p}|N"]]) ** 2)
    se = lambda p, k: 2 * Ca.se(E[Ca.ix[f"{m}|{p}|{k}"]] / Ca.b[Ca.ix[f"{m}|{p}|{k}"]] - E[Ca.ix[f"{m}|{p}|N"]] / Ca.b[Ca.ix[f"{m}|{p}|N"]])
    o = {"W0": R(0, "W"), "W1": R(1, "W"), "se_W0": se(0, "W"), "se_W1": se(1, "W")}
    for p in ("0", "1"):
        if f"{m}|{p}|Z" in Ca.ix: o["Z" + p] = R(int(p), "Z"); o["se_Z" + p] = se(int(p), "Z")
    dz = 1 - o["Z" + zk[m]]; o["pred"] = float(np.log(1 - dz * dU[m] / o["W0"])); o["delta"] = dz
    out[m] = {k: round(v, 3) for k, v in o.items()}
json.dump(out, open("output/results_abs_scale.json", "w"), indent=1); print(out)
