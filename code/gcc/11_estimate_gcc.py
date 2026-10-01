"""Primary and secondary tests of pap/02_pap_GCC.md, sections 5 and 6.

All markets are stacked in one cell regression, so that every ratio, the combination across switches and the
calendar-time Wald test use one joint covariance matrix. Standard errors are clustered two ways (Cameron, Gelbach
and Miller 2011): by calendar week, common to all markets (Saturday-to-Friday blocks), and by firm.

Input   data/clean/gcc_gaps.csv (10_build_gcc.py), data/clean/stock_gaps.csv (TASE, code/04_stocks.py)
Output  output/results_gcc.json, output/tables/gcc_cells.csv
"""
import pandas as pd, numpy as np, json, os
from scipy import stats

def week_id(dates):
    d = pd.to_datetime(dates) + pd.Timedelta(days=2)
    iso = d.dt.isocalendar()
    return iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

def twoway_cov(X, e, g1, g2):
    """Cameron-Gelbach-Miller two-way cluster covariance for OLS: V1 + V2 - V12 (negative eigenvalues clipped)."""
    XtXi = np.linalg.pinv(X.T @ X)
    def meat(groups):
        S = pd.DataFrame(X * e[:, None]).groupby(np.asarray(groups)).sum().values
        return S.T @ S
    g12 = pd.Series(g1).astype(str).values + "|" + pd.Series(g2).astype(str).values
    V = XtXi @ (meat(g1) + meat(g2) - meat(g12)) @ XtXi
    w, Q = np.linalg.eigh((V + V.T) / 2)
    return Q @ np.diag(np.clip(w, 0, None)) @ Q.T

class Cells:
    """Cell means of y by (mk, post, typ), from one regression on cell indicators, with joint two-way covariance."""
    def __init__(self, df, y="z2w"):
        lab = (df.mk + "|" + df.post.astype(str) + "|" + df.typ).values
        self.cats = sorted(set(lab)); self.ix = {c: i for i, c in enumerate(self.cats)}
        X = np.column_stack([(lab == c).astype(float) for c in self.cats])
        yv = df[y].values.astype(float)
        self.b = X.T @ yv / X.sum(0)
        self.V = twoway_cov(X, yv - X @ self.b, df.week.values, df.firm.values)
        self.n = X.sum(0)
        g = df.assign(lab=lab).groupby("lab")
        self.H = g.H.mean().to_dict(); self.U = g.U.mean().to_dict()
        self.firms = df.groupby("mk").firm.nunique().to_dict(); self.weeks = df.groupby("mk").week.nunique().to_dict()
    def has(self, *keys): return all(k in self.ix for k in keys)
    def lr(self, num, den):
        """log(b_num/b_den) and gradient."""
        g = np.zeros(len(self.b)); g[self.ix[num]] += 1 / self.b[self.ix[num]]; g[self.ix[den]] -= 1 / self.b[self.ix[den]]
        return np.log(self.b[self.ix[num]] / self.b[self.ix[den]]), g
    def se(self, g): return float(np.sqrt(g @ self.V @ g))

def market_block(C, m):
    res = dict(cells={}, R={}, firms=int(C.firms[m]), weeks=int(C.weeks[m]))
    for c in C.cats:
        if c.startswith(m + "|"):
            i = C.ix[c]
            res["cells"][c[len(m) + 1:]] = dict(mean=float(C.b[i]), se=float(np.sqrt(C.V[i, i])), n=int(C.n[i]), H=float(C.H[c]), U=float(C.U[c]))
    for p in (0, 1):
        for t in ("W", "Z"):
            k, n = f"{m}|{p}|{t}", f"{m}|{p}|N"
            if C.has(k, n):
                l, g = C.lr(k, n); se = C.se(g); bench = float(np.log(C.H[k] / C.H[n]))
                res["R"][f"{t}_post{p}"] = dict(R=float(np.exp(l)), logR=float(l), se=se, calendar_R=float(np.exp(bench)),
                                                p_one_sided_below_1=float(stats.norm.cdf(l / se)),
                                                p_vs_calendar=float(2 * stats.norm.sf(abs(l - bench) / se)))
    if C.has(*(f"{m}|{p}|{t}" for p in (0, 1) for t in ("W", "N"))):
        l1, g1 = C.lr(f"{m}|1|W", f"{m}|1|N"); l0, g0 = C.lr(f"{m}|0|W", f"{m}|0|N")
        d, g = l1 - l0, g1 - g0; se = C.se(g)
        bench = float(np.log(C.H[f"{m}|1|W"] / C.H[f"{m}|1|N"]) - np.log(C.H[f"{m}|0|W"] / C.H[f"{m}|0|N"]))
        res["DeltaW"] = dict(est=float(d), se=se, p_one_sided=float(stats.norm.cdf(d / se)), calendar_pred=bench,
                             p_vs_calendar=float(2 * stats.norm.sf(abs(d - bench) / se)),
                             U_W_before=float(C.U[f"{m}|0|W"]), U_W_after=float(C.U[f"{m}|1|W"]))
        res["_grad_DeltaW"] = g
    return res

def median_R(df):
    med = df.assign(az=np.sqrt(df.z2)).groupby(["post", "typ"]).az.median()
    return {f"{t}_post{p}": float((med[(p, t)] / med[(p, "N")]) ** 2)
            for p in (0, 1) for t in ("W", "Z") if (p, t) in med.index and (p, "N") in med.index and med[(p, "N")] > 0}

def estimate(df, markets, y="z2w", combine=None):
    """Stack, estimate, and return per-market blocks plus the combination and Wald test over `combine`."""
    C = Cells(df, y)
    out = {m: market_block(C, m) for m in markets}
    if combine:
        grads = [out[m]["_grad_DeltaW"] for m in combine if "DeltaW" in out[m]]
        est = np.array([out[m]["DeltaW"]["est"] for m in combine if "DeltaW" in out[m]])
        G = np.array(grads); S = G @ C.V @ G.T
        w = np.linalg.pinv(S) @ np.ones(len(est)) / (np.ones(len(est)) @ np.linalg.pinv(S) @ np.ones(len(est)))  # GLS weights
        wi = 1 / np.diag(S); wi = wi / wi.sum()                                                                      # inverse-variance
        out["DeltaW_combined"] = dict(est=float(wi @ est), se=float(np.sqrt(wi @ S @ wi)),
                                      gls_est=float(w @ est), gls_se=float(np.sqrt(w @ S @ w)), parts=list(combine),
                                      corr=np.round(S / np.sqrt(np.outer(np.diag(S), np.diag(S))), 3).tolist())
        # joint Wald test of the calendar-time restriction over every W and Z cell of the combined markets
        vec, gr = [], []
        for m in combine:
            for p in (0, 1):
                for t in ("W", "Z"):
                    k, n = f"{m}|{p}|{t}", f"{m}|{p}|N"
                    if C.has(k, n):
                        l, g = C.lr(k, n); vec.append(l - np.log(C.H[k] / C.H[n])); gr.append(g)
        v = np.array(vec); G = np.array(gr); S = G @ C.V @ G.T
        stat = float(v @ np.linalg.pinv(S) @ v)
        out["calendar_wald"] = dict(stat=stat, dof=len(v), p=float(stats.chi2.sf(stat, len(v))))
        out["_C"] = C
    for m in markets: out[m].pop("_grad_DeltaW", None)
    return out

def pooled(P, relative=True):
    """E[y] = a_m + b U + c W with y = z2 relative to the mean ordinary-weeknight z2 of the same market and calendar,
    so that b and c are in weeknight units (the normaliser is treated as known)."""
    base = P[P.typ == "N"].groupby(["mk", "post"]).z2w.mean()
    y = (P.z2w / pd.Series(list(zip(P.mk, P.post))).map(base).values).values if relative else P.z2w.values.astype(float)
    mks = sorted(P.mk.unique()); W = (P.typ == "W").astype(float).values
    X = np.column_stack([(P.mk == m).astype(float).values for m in mks] + [P.U.values, W])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; V = twoway_cov(X, y - X @ beta, P.week.values, P.firm.values)
    se = np.sqrt(np.diag(V))
    return dict(a={m: dict(est=float(beta[i]), se=float(se[i])) for i, m in enumerate(mks)},
                b=dict(est=float(beta[-2]), se=float(se[-2]), p_one_sided=float(stats.norm.sf(beta[-2] / se[-2]))),
                c=dict(est=float(beta[-1]), se=float(se[-1])), n=int(len(P)))

def load():
    g = pd.read_csv("data/clean/gcc_gaps.csv", parse_dates=["date"])
    g = g[g.typ.isin(["N", "W", "Z"])].copy()
    g["week"] = week_id(g.date)
    g["mk"] = g.market.map({"DFMGI": "DFM", "FTFADGI": "ADX", "TASI": "KSA"})
    t = pd.read_csv("data/clean/stock_gaps.csv", parse_dates=["date"])
    t = t.assign(typ=t.typ.replace({"S": "Z"}), firm="TA-" + t.sec.astype(str), z2w=t.z2, week=week_id(t.date),
                 mk=np.where(t.DL == 1, "TASE_dual", "TASE_domestic"), inwin=1)
    return g, t

COLS = ["mk", "firm", "week", "post", "typ", "H", "U", "z2", "z2w", "date"]

def main():
    g, t = load()
    prim = g[g.inwin == 1]
    uae = prim[prim.mk.isin(["DFM", "ADX"])].assign(mk="UAE")
    main_stack = pd.concat([uae[COLS], prim[prim.mk == "KSA"][COLS], t[COLS]], ignore_index=True)
    comb = ["UAE", "KSA", "TASE_domestic", "TASE_dual"]
    out = estimate(main_stack, comb, combine=comb)
    C = out.pop("_C")
    for m in ("DFM", "ADX"):
        out[m] = estimate(prim[prim.mk == m][COLS], [m])[m]
    for m, df in (("UAE", uae), ("KSA", prim[prim.mk == "KSA"]), ("TASE_domestic", t[t.DL == 0]), ("TASE_dual", t[t.DL == 1])):
        out[m]["median_R"] = median_R(df)
    # secondary and robustness, each market estimated on its own
    out["UAE_unwinsorised"] = estimate(uae[COLS], ["UAE"], y="z2")["UAE"]
    out["KSA_unwinsorised"] = estimate(prim[prim.mk == "KSA"][COLS], ["KSA"], y="z2")["KSA"]
    ua = g[g.mk.isin(["DFM", "ADX"])].assign(mk="UAE")
    out["UAE_drop_covid"] = estimate(ua[(ua.inwin == 1) & ~ua.date.between("2020-02-23", "2020-06-30")][COLS], ["UAE"])["UAE"]
    out["UAE_symmetric"] = estimate(ua[ua.date.between("2018-04-01", "2025-09-30")][COLS], ["UAE"])["UAE"]
    ks = g[g.mk == "KSA"]
    out["KSA_pm2"] = estimate(ks[ks.date.between("2011-07-01", "2015-06-30")][COLS], ["KSA"])["KSA"]
    out["KSA_pm5"] = estimate(ks[ks.date.between("2008-07-01", "2018-06-30")][COLS], ["KSA"])["KSA"]
    # TASE under rules closer to the Gulf plan: winsorised at 99.9; and conditional on a non-zero gap
    q = t.z2.quantile(0.999)
    tw = t.assign(z2w=np.minimum(t.z2, q))
    alt = pd.concat([uae[COLS], prim[prim.mk == "KSA"][COLS], tw[COLS]], ignore_index=True)
    r_alt = estimate(alt, comb, combine=comb); r_alt.pop("_C")
    out["alt_TASE_winsorised"] = {k: r_alt[k] for k in ("TASE_domestic", "TASE_dual", "DeltaW_combined", "calendar_wald")}
    tnz = tw[tw.z2 > 0]
    alt2 = pd.concat([uae[COLS], prim[prim.mk == "KSA"][COLS], tnz[COLS]], ignore_index=True)
    r_alt2 = estimate(alt2, comb, combine=comb); r_alt2.pop("_C")
    out["alt_TASE_nonzero"] = {k: r_alt2[k] for k in ("TASE_domestic", "TASE_dual", "DeltaW_combined", "calendar_wald")}
    gulf = estimate(pd.concat([uae[COLS], prim[prim.mk == "KSA"][COLS]]), ["UAE", "KSA"], combine=["UAE", "KSA"]); gulf.pop("_C")
    out["gulf_only"] = {k: gulf[k] for k in ("DeltaW_combined", "calendar_wald")}
    # pooled model, weeknight units
    five = pd.concat([prim[prim.mk.isin(["DFM", "ADX", "KSA"])][COLS], t[COLS]], ignore_index=True)
    out["pooled_prespecified"] = pooled(five, relative=False)   # plan s.6: z2 units, market intercepts
    out["pooled"] = pooled(five)
    out["pooled_gulf"] = pooled(prim[COLS])
    out["pooled_mr"] = pooled(five.assign(mk=five.mk + "_" + five.post.astype(str)))
    os.makedirs("output/tables", exist_ok=True)
    json.dump(out, open("output/results_gcc.json", "w"), indent=1)
    rows = [dict(market=k, cell=c, **v) for k in ("DFM", "ADX", "UAE", "KSA", "TASE_domestic", "TASE_dual") for c, v in out[k]["cells"].items()]
    pd.DataFrame(rows).to_csv("output/tables/gcc_cells.csv", index=False)
    for k in ("DFM", "ADX", "UAE", "KSA", "TASE_domestic", "TASE_dual"):
        print(k, {kk: (round(vv["R"], 3), round(vv["se"], 3)) for kk, vv in out[k]["R"].items()},
              {kk: round(vv, 3) for kk, vv in out[k].get("DeltaW", {}).items() if kk in ("est", "se")})
    for k in ("DeltaW_combined", "calendar_wald", "pooled_prespecified", "pooled", "pooled_gulf", "pooled_mr", "gulf_only", "alt_TASE_winsorised", "alt_TASE_nonzero"):
        v = out[k]
        if k.startswith("pooled"): print(k, v["b"], v["c"])
        elif k.startswith(("gulf", "alt")): print(k, {a: (round(b.get("est", b.get("stat", 0)), 3), round(b.get("se", b.get("p", 0)), 4)) for a, b in v.items() if a in ("DeltaW_combined", "calendar_wald")})
        else: print(k, v)

if __name__ == "__main__":
    main()
