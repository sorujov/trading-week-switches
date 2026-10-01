"""Addendum A1: stock-level triple difference (DL vs non-DL x W vs N x post vs pre)."""
import pandas as pd, numpy as np, json, glob, os, statsmodels.formula.api as smf
rng = np.random.default_rng(7)
B = 4999
comp = pd.read_csv("data/raw/tase/ta125_components.csv", dtype={"sec": str})
DL = set(comp.loc[comp["flags"].fillna("").str.contains("DL"), "sec"])
gi = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date"])[["date", "typ", "post", "H", "U", "Ufut", "g"]].rename(columns={"g": "g_mkt"})

frames = []
for f in sorted(glob.glob("data/raw/tase/stocks/*.csv")):
    sec = os.path.basename(f)[:-4]
    s = pd.read_csv(f, parse_dates=["date"]).sort_values("date").drop_duplicates("date")
    s = s[(s.traded == 1) & (s.open > 0) & (s.base > 0) & (s.close > 0)].copy()
    s["g"] = np.log(s.open / s.base)
    s["d"] = np.log(s.close / s.open)
    s["s22"] = (s.d ** 2).shift(1).rolling(22, min_periods=15).mean()
    s["sec"] = sec
    frames.append(s[["date", "sec", "g", "d", "s22", "turnover"]])
st = pd.concat(frames).merge(gi, on="date", how="inner")
st = st[(st.date >= "2020-03-08") & st.typ.isin(["N", "W", "S"])]
st = st[(st.g.abs() <= 0.25) & (st.s22 > 0)].dropna(subset=["s22"])
st["z2"] = st.g ** 2 / st.s22
st["DL"] = st.sec.isin(DL).astype(int)
st["W"] = (st.typ == "W").astype(int)
st["week"] = (st.date + pd.Timedelta(days=1)).dt.to_period("W").astype(str)
st.to_csv("data/clean/stock_gaps.csv", index=False)

def R_of(df):
    m = df.groupby(["DL", "post", "typ"]).z2.mean()
    return m
def delta_from(m):
    lr = lambda dl, p: np.log(m[(dl, p, "W")] / m[(dl, p, "N")])
    return (lr(1, 1) - lr(1, 0)) - (lr(0, 1) - lr(0, 0)), {f"R_DL{dl}_post{p}": float(np.exp(lr(dl, p))) for dl in (0, 1) for p in (0, 1)}

res = {}
wn = st[st.typ.isin(["W", "N"])]
m = R_of(wn); D, Rs = delta_from(m)
res["n"] = wn.groupby(["DL", "post", "typ"]).size().rename("n").reset_index().to_dict(orient="records")
res["n_stocks"] = dict(DL=int(wn[wn.DL == 1].sec.nunique()), nonDL=int(wn[wn.DL == 0].sec.nunique()))
res["R"] = Rs; res["Delta"] = float(D)
res["S_ratio"] = {f"DL{dl}": float(st[(st.DL == dl) & (st.typ == "S")].z2.mean() / st[(st.DL == dl) & (st.typ == "N") & (st.post == 0)].z2.mean()) for dl in (0, 1)}
# bootstrap: per-week cell sums
cell = wn.groupby(["week", "post", "DL", "typ"]).z2.agg(["sum", "count"]).unstack(["DL", "typ"]).fillna(0)
boots = []
for p in (0, 1):
    c = cell.xs(p, level="post")
    boots.append(c)
def stat_from(sumsc):
    # sumsc: dict period -> Series of (agg, DL, typ)
    mm = {}
    for p in (0, 1):
        sc = sumsc[p]
        for dl in (0, 1):
            for t in ("W", "N"):
                mm[(dl, p, t)] = sc[("sum", dl, t)] / sc[("count", dl, t)]
    return delta_from(mm)[0]
arr = {p: boots[p].values for p in (0, 1)}; cols = boots[0].columns
out = np.empty(B)
for b in range(B):
    sums = {}
    for p in (0, 1):
        A = arr[p]; n = len(A); idx = np.empty(n, int); k = rng.integers(n)
        for j in range(n):
            if j > 0 and rng.random() < 0.25: k = rng.integers(n)
            idx[j] = k; k = (k + 1) % n
        sums[p] = pd.Series(A[idx].sum(0), index=cols)
    out[b] = stat_from(sums)
out = out[np.isfinite(out)]
res["Delta_se"] = float(out.std()); res["Delta_ci"] = list(np.percentile(out, [2.5, 97.5]))
res["Delta_p"] = float(2 * min((out <= 0).mean(), (out >= 0).mean()))
# regression form, week-clustered
wn = wn.assign(Post=wn.post)
mr = smf.ols("z2 ~ W*Post*DL", wn).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(wn.week)[0]})
res["reg"] = dict(coef=mr.params.to_dict(), se=mr.bse.to_dict(), p=mr.pvalues.to_dict(), n=int(mr.nobs))
# log|z| robust version (exploratory A3 analogue)
wz = wn[wn.z2 > 1e-10].assign(lz=lambda x: 0.5 * np.log(x.z2))
ml = smf.ols("lz ~ W*Post*DL + C(sec)", wz).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(wz.week)[0]})
res["reg_logabs"] = {k: dict(coef=ml.params[k], se=ml.bse[k], p=ml.pvalues[k]) for k in ml.params.index if not k.startswith("C(")}
# market-adjusted (idiosyncratic) gap: residual from stock-specific regression on index gap, estimated on N gaps pre
def idio(df):
    base = df[(df.typ == "N") & (df.post == 0)]
    if len(base) < 100: return pd.Series(np.nan, index=df.index)
    b = np.polyfit(base.g_mkt, base.g, 1)
    return (df.g - (b[0] * df.g_mkt + b[1])) ** 2 / df.s22
wn = wn.copy(); wn["iz2"] = wn.groupby("sec", group_keys=False).apply(idio)
mi = wn.dropna(subset=["iz2"]).groupby(["DL", "post", "typ"]).iz2.mean()
Di, Ri = delta_from(mi); res["idio"] = dict(Delta=float(Di), R=Ri)
json.dump(res, open("output/results_stocks.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
