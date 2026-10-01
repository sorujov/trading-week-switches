"""Further analyses of the Tel Aviv share panel (all exploratory, not in plan 01)."""
import pandas as pd, numpy as np, json, statsmodels.formula.api as smf
from datetime import datetime, time
from zoneinfo import ZoneInfo
rng = np.random.default_rng(3)
IL, NY, LDN = ZoneInfo("Asia/Jerusalem"), ZoneInfo("America/New_York"), ZoneInfo("Europe/London")
out = {}
st = pd.read_csv("data/clean/stock_gaps.csv", parse_dates=["date"], dtype={"sec": str})
comp = pd.read_csv("data/raw/tase/ta125_components.csv", dtype={"sec": str}).set_index("sec")

# 1) zero gaps: participation vs variance margins
st["nz"] = (st.g != 0).astype(int)
rows = []
for (dl, p, t), x in st.groupby(["DL", "post", "typ"]):
    rows.append(dict(DL=dl, post=p, typ=t, n=len(x), p_nonzero=x.nz.mean(), mean_z2=x.z2.mean(), mean_z2_nz=x.loc[x.nz == 1, "z2"].mean()))
Z = pd.DataFrame(rows)
for (dl, p), x in Z.groupby(["DL", "post"]):
    n = x[x.typ == "N"].iloc[0]
    Z.loc[x.index, "R_uncond"] = x.mean_z2 / n.mean_z2
    Z.loc[x.index, "R_cond"] = x.mean_z2_nz / n.mean_z2_nz
    Z.loc[x.index, "P_ratio"] = x.p_nonzero / n.p_nonzero
Z.to_csv("output/tables/rev_zero_margins.csv", index=False)
out["zero_margins"] = Z.to_dict(orient="records")

# 2) firm-level Sunday and weekend ratios for DL firms
pre = st[st.post == 0]
fr = []
for sec, x in st[st.DL == 1].groupby("sec"):
    xp = x[x.post == 0]; xq = x[x.post == 1]
    nN = xp[xp.typ == "N"].z2.mean()
    fr.append(dict(sec=sec, symbol=comp.loc[sec, "symbol"], n_pre=len(xp),
                   R_S=xp[xp.typ == "S"].z2.mean() / nN if len(xp) else np.nan,
                   R_W_pre=xp[xp.typ == "W"].z2.mean() / nN if len(xp) else np.nan,
                   R_W_post=xq[xq.typ == "W"].z2.mean() / xq[xq.typ == "N"].z2.mean() if len(xq) else np.nan,
                   p_nonzero=x.nz.mean()))
F = pd.DataFrame(fr).sort_values("R_S")
F.to_csv("output/tables/rev_dl_firms.csv", index=False)
out["dl_firms"] = F.to_dict(orient="records")
out["dl_firms_median_R_S"] = float(F.R_S.median())
out["dl_firms_share_R_S_below_1"] = float((F.R_S < 1).mean())

# 3) two-way clustered (week x firm) SEs; exclusions
def twoway(formula, df):
    g = np.column_stack([pd.factorize(df.week)[0], pd.factorize(df.sec)[0]])
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": g})
sn = pre[pre.typ.isin(["N", "S"])].assign(S=lambda x: (x.typ == "S").astype(int))
res = {}
for lab, df in [("all", sn), ("excl_ENOG", sn[sn.sec != "1155290"]),
                ("excl_ENOG_thin", sn[~sn.sec.isin(comp.index[comp["flags"].fillna("").str.contains("DL\\|MM")])]),
                ("nonzero_only", sn[sn.nz == 1])]:
    m = twoway("z2 ~ S*DL", df)
    res[lab] = dict(coef=m.params["S:DL"], se=m.bse["S:DL"], p=m.pvalues["S:DL"], n=int(m.nobs))
out["sunday_DL_twoway"] = res
wn = st[st.typ.isin(["W", "N"])]
m = twoway("z2 ~ W*post*DL", wn)
out["triple_twoway"] = dict(coef=m.params["W:post:DL"], se=m.bse["W:post:DL"], p=m.pvalues["W:post:DL"])
m = twoway("z2 ~ W*post*DL", wn[wn.nz == 1])
out["triple_twoway_nonzero"] = dict(coef=m.params["W:post:DL"], se=m.bse["W:post:DL"], p=m.pvalues["W:post:DL"])

# 4) direct test: DL gap return on own foreign-listing return realised inside the gap
tick = {"629014": "TEVA", "1081124": "ESLT", "1082379": "TSEM", "1084557": "NVMI", "720011": "ENLT", "273011": "NICE",
        "1134402": "ORA", "1095264": "CAMT", "281014": "ICL", "1134139": "KEN", "1155290": "ENOG.L", "256016": "FORTY",
        "1082510": "GILT", "1175116": "NYAX", "1129543": "OPK"}
gi = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date", "prev"])[["date", "prev"]]
def tase_close(d):
    wd = d.dayofweek
    if wd == 6: return time(15, 50)
    if wd == 4: return time(13, 45)
    return time(17, 25)
gw = {}
for _, r in gi.iterrows():
    gw[r.date] = (datetime.combine(r.prev.date(), tase_close(r.prev), IL).timestamp(), datetime.combine(r.date.date(), time(10, 0), IL).timestamp())
fx_rows = []
for sec, tk in tick.items():
    f = pd.read_csv(f"data/raw/tase/foreign_{tk}.csv", index_col=0, parse_dates=True)
    tz, ct = (LDN, time(16, 30)) if tk.endswith(".L") else (NY, time(16, 0))
    lr = np.log(f["Close"]).diff()
    closes = np.array([datetime.combine(d.date(), ct, tz).timestamp() for d in f.index])
    x = st[st.sec == sec]
    for _, r in x.iterrows():
        a, b = gw[r.date]
        idx = np.where((closes > a) & (closes <= b))[0]
        fx_rows.append(dict(sec=sec, date=r.date, typ=r.typ, post=r.post, g=r.g, z2=r.z2, s22=r.s22,
                            n_f=len(idx), rf=float(lr.iloc[idx].sum()) if len(idx) else 0.0))
FX = pd.DataFrame(fx_rows).dropna()
lo = []
for (p, t), x in FX.groupby(["post", "typ"]):
    m = smf.ols("g ~ rf", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.date)[0]}) if x.rf.abs().sum() > 0 else None
    resid = (x.g - (m.params["Intercept"] + m.params["rf"] * x.rf)) if m is not None else (x.g - x.g.mean())
    lo.append(dict(post=p, typ=t, n=len(x), mean_nf=x.n_f.mean(), beta=m.params["rf"] if m is not None else np.nan,
                   se=m.bse["rf"] if m is not None else np.nan, r2=m.rsquared if m is not None else 0.0,
                   mean_z2=x.z2.mean(), mean_resid_z2=(resid ** 2 / x.s22).mean()))
LO = pd.DataFrame(lo)
for p, x in LO.groupby("post"):
    n = x[x.typ == "N"].iloc[0]
    LO.loc[x.index, "R"] = x.mean_z2 / n.mean_z2
    LO.loc[x.index, "R_resid"] = x.mean_resid_z2 / n.mean_resid_z2
LO.to_csv("output/tables/rev_dl_foreign_loadings.csv", index=False)
out["dl_foreign_loadings"] = LO.to_dict(orient="records")

# 5) index close-to-close counterpart and intraday variance by weekday
g = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date"])
off = pd.read_csv("data/raw/tase/tase_TA125_official.csv", parse_dates=["date"])
off["cc"] = np.log(off.close).diff(); off["d"] = np.log(off.close / off.open)
off["v22"] = (off.cc ** 2).shift(1).rolling(22, min_periods=15).mean()
g = g.merge(off[["date", "v22"]], on="date")
g = g[(g.date >= "2020-03-08") & g.typ.isin(["N", "W", "S"])].dropna(subset=["v22"])
g["c2"] = g.cc ** 2 / g.v22
g["week"] = (g.date + pd.Timedelta(days=1)).dt.to_period("W").astype(str)
def ratio(df, col): return df.loc[df.typ == "W", col].mean() / df.loc[df.typ == "N", col].mean()
cc = {}
for p in (0, 1):
    df = g[g.post == p]; ws = df.week.unique(); grp = {w: x for w, x in df.groupby("week")}
    bs = [ratio(pd.concat([grp[w] for w in rng.choice(ws, len(ws))]), "c2") for _ in range(1999)]
    cc[p] = (ratio(df, "c2"), np.log(bs))
out["cc_ratio"] = dict(R_pre=cc[0][0], R_post=cc[1][0], logdiff=float(np.log(cc[1][0] / cc[0][0])),
                       se=float(np.std(cc[1][1] - cc[0][1])), ci=list(np.percentile(cc[1][1] - cc[0][1], [2.5, 97.5])),
                       R_S_pre=float(g[(g.post == 0) & (g.typ == "S")].c2.mean() / g[(g.post == 0) & (g.typ == "N")].c2.mean()))
off2 = off[off.date >= "2020-03-08"].copy(); off2["post"] = (off2.date >= "2026-01-05").astype(int)
out["intraday_var_by_weekday"] = (off2.groupby(["post", off2.date.dt.day_name()]).d.apply(lambda s: (s ** 2).mean() * 1e4)).round(3).rename_axis(["post", "dow"]).reset_index().to_dict(orient="records")

# 6) rolling placebo: log R in consecutive non-overlapping blocks of 34 weekends (pre period)
gi2 = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date"])
gp = gi2[(gi2.date >= "2020-03-08") & (gi2.post == 0) & gi2.typ.isin(["N", "W"])].dropna(subset=["z2"]).sort_values("date")
wdates = gp[gp.typ == "W"].date.values
lr = []
for k in range(0, len(wdates) - 33, 34):
    a, b = wdates[k], wdates[k + 33]
    blk = gp[(gp.date >= a) & (gp.date <= b + np.timedelta64(4, "D"))]
    lr.append(np.log(ratio(blk, "z2")))
lr = np.array(lr)
out["rolling_blocks"] = dict(n_blocks=len(lr), logR=list(lr), sd_diff_consecutive=float(np.std(np.diff(lr), ddof=1)), sd_logR=float(np.std(lr, ddof=1)))

# 7) vendor details: unmatched Yahoo rows and Emaar weekday counts
lag = pd.read_csv("output/tables/vendor_lag_by_year.csv", index_col=0)
out["yahoo_unmatched_rows"] = {int(y): int(r.n_yahoo - r.matched) for y, r in lag.loc[2019:2025].iterrows()}
em = pd.read_csv("data/raw/tase/EMAAR.AE.csv", index_col=0, parse_dates=True)
em = em[em.index >= "2022-01-03"]
out["emaar_weekday_rows_since_2022"] = em.index.day_name().value_counts().to_dict()
R = json.load(open("output/results.json"))
out["BF_N"] = R["BF_N"]; out["BF_W"] = R["BF_W"]
json.dump(out, open("output/results_revision.json", "w"), indent=1, default=float)
print(Z.round(3).to_string()); print(F.round(2).to_string()); print(LO.round(3).to_string())
for k in ["dl_firms_median_R_S", "dl_firms_share_R_S_below_1", "sunday_DL_twoway", "triple_twoway", "triple_twoway_nonzero", "cc_ratio", "rolling_blocks", "yahoo_unmatched_rows", "emaar_weekday_rows_since_2022", "BF_N"]:
    print(k, json.dumps(out[k], default=float))
print(pd.DataFrame(out["intraday_var_by_weekday"]))

# 8) time-varying foreign listing: DL active only from the first date the foreign listing has prices
first = {}
for sec, tk in tick.items():
    f = pd.read_csv(f"data/raw/tase/foreign_{tk}.csv", index_col=0, parse_dates=True)
    first[sec] = f.index.min()
st2 = st.copy()
st2["DLa"] = [int(s in first and d >= first[s]) for s, d in zip(st2.sec, st2.date)]
st2 = st2[~((st2.DL == 1) & (st2.DLa == 0))]          # drop DL firms' observations before their foreign listing existed
st2 = st2[st2.sec != "1236736"]                        # CYBR: no foreign price file (no pre-switch TASE data either)
sn2 = st2[(st2.post == 0) & st2.typ.isin(["N", "S"])].assign(S=lambda x: (x.typ == "S").astype(int))
m = twoway("z2 ~ S*DL", sn2)
c2 = st2[(st2.post == 0)].groupby(["DL", "typ"]).z2.mean()
out["active_DL"] = dict(first_foreign={k: str(v.date()) for k, v in first.items()}, coef=m.params["S:DL"], se=m.bse["S:DL"], p=m.pvalues["S:DL"],
                        R_S_DL=float(c2[(1, "S")] / c2[(1, "N")]), R_S_dom=float(c2[(0, "S")] / c2[(0, "N")]))
json.dump(out, open("output/results_revision.json", "w"), indent=1, default=float)
print("active_DL", json.dumps(out["active_DL"], default=float))

# 9) firm-level randomisation inference for the Sunday-night x DL contrast:
#    reassign the DL label to 15 random firms (the number of DL firms with pre-switch data), 4,999 times
rng2 = np.random.default_rng(99)
snr = pre[pre.typ.isin(["N", "S"])]
cellm = snr.groupby(["sec", "typ"]).z2.agg(["sum", "count"]).unstack("typ").fillna(0)
secs = cellm.index.values; isdl = np.isin(secs, list(set(st.loc[st.DL == 1, "sec"])))
A = cellm.values  # columns: sum N, sum S, count N, count S
cols = list(cellm.columns)
iN, iS, cN, cS = cols.index(("sum", "N")), cols.index(("sum", "S")), cols.index(("count", "N")), cols.index(("count", "S"))
def stat(mask):
    d, o = A[mask].sum(0), A[~mask].sum(0)
    return np.log((d[iS] / d[cS]) / (d[iN] / d[cN])) - np.log((o[iS] / o[cS]) / (o[iN] / o[cN]))
obs = stat(isdl); k = int(isdl.sum())
perm = np.array([stat(np.isin(np.arange(len(secs)), rng2.choice(len(secs), k, replace=False))) for _ in range(4999)])
out["sunday_randomisation"] = dict(obs_logdiff=float(obs), k=k, n_firms=len(secs), p_one_sided=float((perm <= obs).mean()),
                                   perm_q01=float(np.quantile(perm, 0.01)), perm_q05=float(np.quantile(perm, 0.05)))
json.dump(out, open("output/results_revision.json", "w"), indent=1, default=float)
print("randomisation", json.dumps(out["sunday_randomisation"]))

# 10) stock-level close-to-close counterpart (weekend information priced during the first session)
import glob, os
gi3 = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date"])[["date", "typ", "post"]]
fr2 = []; anch = {}
for f in sorted(glob.glob("data/raw/tase/stocks/*.csv")):
    sec = os.path.basename(f)[:-4]
    s = pd.read_csv(f, parse_dates=["date"]).sort_values("date").drop_duplicates("date")
    s = s[(s.traded == 1) & (s.base > 0) & (s.close > 0)].copy()
    s["cc"] = np.log(s.close / s.base)
    s["v22"] = (s.cc ** 2).shift(1).rolling(22, min_periods=15).mean()
    s["prevclose"] = s.close.shift(1)
    anch[sec] = float(((s.base / s.prevclose - 1).abs() > 1e-6).mean())
    s["sec"] = sec
    fr2.append(s[["date", "sec", "cc", "v22"]])
sc = pd.concat(fr2).merge(gi3, on="date")
sc = sc[(sc.date >= "2020-03-08") & sc.typ.isin(["N", "S", "W"]) & (sc.cc.abs() <= 0.25) & (sc.v22 > 0)].dropna()
dlset = set(st.loc[st.DL == 1, "sec"])
sc["DL"] = sc.sec.isin(dlset).astype(int); sc["c2"] = sc.cc ** 2 / sc.v22
sc["week"] = (sc.date + pd.Timedelta(days=1)).dt.to_period("W").astype(str)
cm = sc.groupby(["DL", "post", "typ"]).c2.mean()
Rcc = {f"DL{d}_post{p}_{t}": float(cm[(d, p, t)] / cm[(d, p, "N")]) for (d, p, t) in cm.index}
wnc = sc[sc.typ.isin(["N", "W"])].assign(W=lambda x: (x.typ == "W").astype(int))
m = twoway("c2 ~ W*post*DL", wnc)
snc = sc[(sc.post == 0) & sc.typ.isin(["N", "S"])].assign(S=lambda x: (x.typ == "S").astype(int))
m2 = twoway("c2 ~ S*DL", snc)
lrr = lambda d, p: np.log(cm[(d, p, "W")] / cm[(d, p, "N")])
out["stock_cc"] = dict(R=Rcc, triple_logdiff=float((lrr(1, 1) - lrr(1, 0)) - (lrr(0, 1) - lrr(0, 0))),
                       triple_reg=dict(coef=m.params["W:post:DL"], se=m.bse["W:post:DL"], p=m.pvalues["W:post:DL"]),
                       sunday_reg=dict(coef=m2.params["S:DL"], se=m2.bse["S:DL"], p=m2.pvalues["S:DL"]))
dl_anch = {k: v for k, v in anch.items() if k in dlset}
out["base_anchor_share"] = dict(dl_max=float(max(dl_anch.values())), dl_mean=float(np.mean(list(dl_anch.values()))), all_max=float(max(anch.values())))
# levels for framing
lv = st[st.post == 0].groupby(["DL", "typ"]).z2.mean()
out["levels_pre"] = {f"DL{d}_{t}": float(v) for (d, t), v in lv.items()}
json.dump(out, open("output/results_revision.json", "w"), indent=1, default=float)
print("stock_cc", json.dumps(out["stock_cc"], default=float)); print("anchor", out["base_anchor_share"]); print("levels", out["levels_pre"])
