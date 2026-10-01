"""Pre-registered estimation (01_pap.md, commit ef4f235). Writes output/tables/*.csv|tex and output/results.json."""
import pandas as pd, numpy as np, json, statsmodels.api as sm
from scipy import stats
rng = np.random.default_rng(20260923)
B = 9999

def load(name, start="2020-03-08", end="2026-09-23", excl=()):
    g = pd.read_csv(f"data/clean/gaps_{name}.csv", parse_dates=["date", "prev"])
    g = g[(g.date >= start) & (g.date <= end)].dropna(subset=["g", "s22"]).copy()
    for a, b in excl:
        g = g[~((g.date >= a) & (g.date <= b))]
    # week id: pre weeks run Sun-Thu, post Mon-Fri; shifting by +1 day puts both in one ISO week
    g["week"] = (g.date + pd.Timedelta(days=1)).dt.to_period("W").astype(str)
    return g

def block_boot_ratio(df, col, B=B, mean_block=4):
    """Stationary block bootstrap over weeks of R = mean(col|W)/mean(col|N)."""
    weeks = df["week"].unique(); nW = len(weeks)
    by = {w: x for w, x in df.groupby("week")}
    arrW = {w: (x.loc[x.typ == "W", col].sum(), (x.typ == "W").sum(), x.loc[x.typ == "N", col].sum(), (x.typ == "N").sum()) for w, x in by.items()}
    A = np.array([arrW[w] for w in weeks])
    out = np.empty(B); p = 1 / mean_block
    for b in range(B):
        idx = np.empty(nW, dtype=int); k = rng.integers(nW)
        for j in range(nW):
            if j > 0 and rng.random() < p: k = rng.integers(nW)
            idx[j] = k; k = (k + 1) % nW
        s = A[idx].sum(0)
        out[b] = (s[0] / s[1]) / (s[2] / s[3]) if s[1] > 0 and s[3] > 0 else np.nan
    return out

def ratio(df, col):
    return df.loc[df.typ == "W", col].mean() / df.loc[df.typ == "N", col].mean()

def p1(g, col="z2", label="primary"):
    pre, post = g[g.post == 0], g[g.post == 1]
    Rpre, Rpost = ratio(pre, col), ratio(post, col)
    bpre, bpost = block_boot_ratio(pre, col), block_boot_ratio(post, col)
    ld = np.log(bpost) - np.log(bpre)
    ld = ld[np.isfinite(ld)]
    est = np.log(Rpost) - np.log(Rpre)
    se = ld.std()
    p_boot = 2 * min((ld <= 0).mean(), (ld >= 0).mean())
    ci = np.percentile(ld, [2.5, 97.5])
    # regression form with HAC
    sub = g[g.typ.isin(["N", "W", "S"])].copy()
    sub["W"] = (sub.typ == "W").astype(int); sub["S"] = (sub.typ == "S").astype(int)
    sub["WxPost"] = sub.W * sub.post
    X = sm.add_constant(sub[["W", "post", "WxPost", "S"]])
    m = sm.OLS(sub[col], X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    # Brown-Forsythe on W gaps pre vs post relative to N (levels of g normalised)
    return dict(label=label, col=col, n_W_pre=int((pre.typ == "W").sum()), n_N_pre=int((pre.typ == "N").sum()),
                n_S_pre=int((pre.typ == "S").sum()), n_W_post=int((post.typ == "W").sum()), n_N_post=int((post.typ == "N").sum()),
                R_pre=Rpre, R_post=Rpost, R_pre_ci=list(np.nanpercentile(bpre, [2.5, 97.5])), R_post_ci=list(np.nanpercentile(bpost, [2.5, 97.5])),
                R_S_pre=pre.loc[pre.typ == "S", col].mean() / pre.loc[pre.typ == "N", col].mean(),
                logdiff=est, logdiff_se=se, logdiff_ci=list(ci), p_boot=p_boot,
                mde_log=2.8 * se,
                reg=dict(coef=m.params.to_dict(), se=m.bse.to_dict(), p=m.pvalues.to_dict(), n=int(m.nobs)))

def p2(g, col="z2"):
    sub = g[g.typ.isin(["N", "W", "S"])].copy()
    X = sm.add_constant(sub[["H", "U"]])
    m = sm.OLS(sub[col], X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    b = m.params
    # shares for a pre weekend gap
    wpre = sub[(sub.typ == "W") & (sub.post == 0)][["H", "U"]].mean()
    wpost = sub[(sub.typ == "W") & (sub.post == 1)][["H", "U"]].mean()
    n = sub[sub.typ == "N"][["H", "U"]].mean()
    # predicted ratios
    pred = lambda v: b["const"] + b["H"] * v["H"] + b["U"] * v["U"]
    # hours-only and U-only restricted models (information criteria)
    mH = sm.OLS(sub[col], sm.add_constant(sub[["H"]])).fit(); mU = sm.OLS(sub[col], sm.add_constant(sub[["U"]])).fit()
    mf = sm.OLS(sub[col], X).fit()
    # structural model by gap-type means (robust to heavy tails): fit on cell means
    return dict(coef=b.to_dict(), se=m.bse.to_dict(), p=m.pvalues.to_dict(), n=int(m.nobs), r2=mf.rsquared,
                pred_R_pre=pred(wpre) / pred(n), pred_R_post=pred(wpost) / pred(n),
                share_hours_Wpre=b["H"] * wpre["H"] / pred(wpre), share_U_Wpre=b["U"] * wpre["U"] / pred(wpre),
                aic=dict(H_only=mH.aic, U_only=mU.aic, both=mf.aic), cells=dict(Wpre=wpre.to_dict(), Wpost=wpost.to_dict(), N=n.to_dict()))

def p3(g):
    out = {}
    sub = g[g.typ.isin(["N", "W", "S"])].copy()
    res = {}
    for (post, typ), x in sub.groupby(["post", "typ"]):
        cols = [c for c in ["us_thu", "us_fri", "us_other"] if x[c].abs().sum() > 0]
        if not cols:
            res[f"{typ}_{post}"] = dict(n=len(x), coef={}, se={}, r2=0.0); x = x.assign(resid=x.g - x.g.mean())
        else:
            m = sm.OLS(x.g, sm.add_constant(x[cols])).fit(cov_type="HC1")
            res[f"{typ}_{post}"] = dict(n=len(x), coef=m.params.to_dict(), se=m.bse.to_dict(), r2=m.rsquared)
            x = x.assign(resid=m.resid)
        sub.loc[x.index, "resid"] = x.resid
    sub["rz2"] = sub.resid ** 2 / sub.s22
    out["loadings"] = res
    out["resid_P1"] = p1(sub, "rz2", "residual after US projection")
    return out

def p4(g):
    r = {}
    for post in (0, 1):
        for typ in ("W", "N"):
            x = g[(g.post == post) & (g.typ == typ)]
            r[f"{typ}_{post}"] = dict(n=len(x), mean_g=x.g.mean() * 100, se_g=x.g.std() / np.sqrt(len(x)) * 100,
                                      mean_cc=x.cc.mean() * 100, se_cc=x.cc.std() / np.sqrt(len(x)) * 100)
    return r

if __name__ == "__main__":
    import os; os.makedirs("output/tables", exist_ok=True)
    R = {}
    g = load("TA125")
    R["P1"] = p1(g)
    R["P1_levels"] = p1(g.assign(g2=g.g ** 2 * 1e4), "g2", "levels")
    # Brown-Forsythe: W-pre vs W-post normalised gap
    zpre = g[(g.post == 0) & (g.typ == "W")].eval("g/sqrt(s22)"); zpost = g[(g.post == 1) & (g.typ == "W")].eval("g/sqrt(s22)")
    npre = g[(g.post == 0) & (g.typ == "N")].eval("g/sqrt(s22)"); npost = g[(g.post == 1) & (g.typ == "N")].eval("g/sqrt(s22)")
    R["BF_W"] = dict(zip(["stat", "p"], stats.levene(zpre, zpost, center="median")))
    R["BF_N"] = dict(zip(["stat", "p"], stats.levene(npre, npost, center="median")))
    R["P2"] = p2(g)
    R["P3"] = p3(g)
    R["P4"] = p4(g)
    # robustness
    rob = {}
    rob["TA35"] = p1(load("TA35"), label="TA-35")
    rob["s66"] = p1(g.dropna(subset=["z2_66"]), "z2_66", "66-session normaliser")
    rob["excl2020"] = p1(load("TA125", start="2021-01-01"), label="excl. 2020")
    rob["excl_war23"] = p1(load("TA125", excl=[("2023-10-07", "2023-12-31")]), label="excl. Oct-Dec 2023")
    rob["excl_jun25"] = p1(load("TA125", excl=[("2025-06-01", "2025-06-30")]), label="excl. June 2025")
    gl = load("TA125", start="2005-01-02"); rob["long_pre"] = p1(gl, label="pre sample from 2005")
    lo, hi = g.z2.quantile([.01, .99]); rob["winsor"] = p1(g.assign(zw=g.z2.clip(lo, hi)), "zw", "winsorised 1/99")
    # exclude gaps adjacent to holidays: previous or next gap is type H
    gh = g.sort_values("date").copy(); hflag = (gh.typ == "H")
    adj = hflag.shift(1, fill_value=False) | hflag.shift(-1, fill_value=False)
    rob["excl_hol_adj"] = p1(gh[~adj], label="excl. holiday-adjacent")
    # placebo: pretend switch on 2025-01-05 using pre data 2020-03-08..2025-12-31
    gp = load("TA125", end="2025-12-31").copy(); gp["post"] = (gp.date >= "2025-01-05").astype(int)
    rob["placebo_2025"] = p1(gp, label="placebo switch 2025-01-05")
    R["robust"] = rob
    json.dump(R, open("output/results.json", "w"), indent=1, default=float)
    # print summary
    def s(r): return f"{r['label']:32s} Rpre={r['R_pre']:.2f} Rpost={r['R_post']:.2f} RSpre={r['R_S_pre']:.2f} logdiff={r['logdiff']:.3f} (se {r['logdiff_se']:.3f}, p={r['p_boot']:.4f}) nWpost={r['n_W_post']} delta={r['reg']['coef']['WxPost']:.3f} p_reg={r['reg']['p']['WxPost']:.4f}"
    print(s(R["P1"])); print(s(R["P1_levels"]))
    for k, v in rob.items(): print(s(v))
    print("BF", R["BF_W"], R["BF_N"])
    print("P2", json.dumps({k: R["P2"][k] for k in ["coef", "se", "p", "pred_R_pre", "pred_R_post", "share_hours_Wpre", "aic"]}, default=float, indent=0))
    print("P3 loadings", json.dumps(R["P3"]["loadings"], default=float))
    print(s(R["P3"]["resid_P1"]))
    print("P4", json.dumps(R["P4"], default=float))
