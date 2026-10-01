"""Exploratory analyses of pap/03_addendum_B_exploratory.md (B1-B6). Labelled exploratory in the paper.

Input   data/clean/gcc_gaps.csv, data/clean/stock_gaps.csv, output/results_gcc.json, raw LSEG stock files
Output  output/results_gcc_exploratory.json
"""
import pandas as pd, numpy as np, json, glob, os, importlib.util
from scipy import stats
spec = importlib.util.spec_from_file_location("est", "code/gcc/11_estimate_gcc.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)

class _E:  # thin wrapper: one market estimated on its own, calendar-week and firm clusters
    @staticmethod
    def market_tests(df, y="z2w"):
        d = df.assign(mk="X", week=est.week_id(df.date))
        return est.estimate(d[est.COLS], ["X"], y=y)["X"]
mt = _E

RAMADAN = ["2008-09-01", "2009-08-22", "2010-08-11", "2011-08-01", "2012-07-20", "2013-07-09", "2014-06-28", "2015-06-18",
           "2016-06-06", "2017-05-27", "2018-05-16", "2019-05-06", "2020-04-24", "2021-04-13", "2022-04-02", "2023-03-23",
           "2024-03-11", "2025-03-01", "2026-02-18"]   # first day (approx., +/-1 day); 30 days assumed

def in_ramadan(d):
    m = np.zeros(len(d), bool)
    for s in RAMADAN:
        s = pd.Timestamp(s); m |= (d >= s) & (d < s + pd.Timedelta(days=30))
    return m

def summary(r):
    o = {k: dict(R=round(v["R"], 3), se=round(v["se"], 3)) for k, v in r["R"].items()}
    if "DeltaW" in r: o["DeltaW"] = dict(est=round(r["DeltaW"]["est"], 3), se=round(r["DeltaW"]["se"], 3))
    return o

def main():
    res = json.load(open("output/results_gcc.json"))
    g = pd.read_csv("data/clean/gcc_gaps.csv", parse_dates=["date", "prev"])
    g = g[g.typ.isin(["N", "W", "Z"])]
    P = g[g.inwin == 1]
    mk = {"UAE": P[P.market.isin(["DFMGI", "FTFADGI"])], "KSA": P[P.market == "TASI"]}
    out = {}
    # ---- B1 bound: weeknight delta = 1 - R_Z (regime with a clean Z), weekend delta_W implied by the switch
    t = pd.read_csv("data/clean/stock_gaps.csv", parse_dates=["date"])
    b1 = {}
    for name, zkey in (("UAE", "Z_post1"), ("KSA", "Z_post0"), ("TASE_domestic", "Z_post0"), ("TASE_dual", "Z_post0")):
        r = res[name]; D = r["DeltaW"]; Rw0 = r["R"]["W_post0"]["R"]; dU = D["U_W_before"] - D["U_W_after"]
        dW = Rw0 * (1 - np.exp(D["est"])) / dU
        dW_se = Rw0 * np.exp(D["est"]) * D["se"] / dU
        dz = 1 - r["R"][zkey]["R"]; dz_se = r["R"][zkey]["R"] * r["R"][zkey]["se"]
        b1[name] = dict(delta_weeknight=dz, se=dz_se, delta_weekend=dW, se_w=dW_se, dU=dU,
                        pred_DeltaW_from_weeknight=float(np.log(1 - dz * dU / Rw0)), obs_DeltaW=D["est"], obs_se=D["se"])
    # combined upper bound on the weekend per-session effect (inverse-variance)
    order = res["DeltaW_combined"]["parts"]; Rc = np.array(res["DeltaW_combined"]["corr"])
    sw = np.array([b1[k]["se_w"] for k in order]); e = np.array([b1[k]["delta_weekend"] for k in order])
    S = Rc * np.outer(sw, sw); w = (1 / sw ** 2) / (1 / sw ** 2).sum()
    comb = float(w @ e); cse = float(np.sqrt(w @ S @ w))   # inverse-variance weights, joint covariance (conditional on R_W before)
    out["B1"] = dict(markets=b1, delta_weekend_combined=float(comb), se=float(cse), upper95=float(comb + 1.645 * cse))
    # ---- KSA before the switch: Saturday nights vs Sunday nights (both without a US session)
    k0 = mk["KSA"][mk["KSA"].post == 0].copy()
    k0["typ"] = np.where(k0.typ == "Z", np.where(k0.prev.dt.dayofweek == 5, "Zsat", np.where(k0.prev.dt.dayofweek == 6, "Zsun", "Zoth")), k0.typ)
    C = est.Cells(k0.assign(mk="K", week=est.week_id(k0.date)), "z2w")
    split = {}
    for t_ in ("Zsat", "Zsun"):
        l, gv = C.lr(f"K|0|{t_}", "K|0|N"); split[t_] = dict(R=float(np.exp(l)), se=C.se(gv), n=int(C.n[C.ix[f"K|0|{t_}"]]))
    l1, g1 = C.lr("K|0|Zsat", "K|0|Zsun"); split["log_sat_minus_sun"] = dict(est=float(l1), se=C.se(g1))
    out["KSA_sat_sun_split"] = split
    # ---- B2 US holidays
    def hol_filter(df):
        md = df.date.dt.strftime("%m-%d")
        season = (md >= "12-24") | (md <= "01-02")
        return df[~season]
    out["B2_no_holiday_season"] = {k: summary(mt.market_tests(hol_filter(v))) for k, v in mk.items()}
    uae_post = mk["UAE"][mk["UAE"].post == 1]
    # Monday US holidays: the closure from Monday close to Tuesday open (prev is Monday)
    zm = uae_post[(uae_post.typ.isin(["N"])) | ((uae_post.typ == "Z") & (uae_post.prev.dt.dayofweek == 0))]
    out["B2_UAE_monday_US_holidays"] = summary(mt.market_tests(pd.concat([zm, uae_post[uae_post.typ == "W"]])))
    # ---- B3 UAE Sunday nights
    u0 = mk["UAE"][mk["UAE"].post == 0].copy()
    # (i) session characteristics by weekday, from raw files (pre-2022, window)
    rows = []
    for f in glob.glob("data/raw/lseg/job02/stocks/DFMGI__*.csv") + glob.glob("data/raw/lseg/job02/stocks/FTFADGI__*.csv"):
        firm = os.path.basename(f).split("__")[1][:-4]
        if firm not in set(u0.firm): continue
        s = pd.read_csv(f, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
        s = s[(s.index >= "2018-01-02") & (s.index < "2022-01-01")]
        med = s.ACVOL_UNS.median()
        rows.append(pd.DataFrame(dict(firm=firm, wd=s.index.day_name().str[:3], relvol=s.ACVOL_UNS / med,
                                      zero_intraday=(s.OPEN_PRC == s.TRDPRC_1).astype(float),
                                      d2=np.log(s.TRDPRC_1 / s.OPEN_PRC) ** 2, date=s.index)))
    S = pd.concat(rows)
    out["B3_session_by_weekday"] = S.groupby("wd").agg(relvol_mean=("relvol", "mean"), relvol_median=("relvol", "median"),
                                                       zero_intraday=("zero_intraday", "mean"), d2=("d2", "mean")).round(5).to_dict(orient="index")
    # (ii) reversal: g_t on preceding session's intraday return d_{t-1}, by night type
    lag = {}
    for f in glob.glob("data/raw/lseg/job02/stocks/DFMGI__*.csv") + glob.glob("data/raw/lseg/job02/stocks/FTFADGI__*.csv"):
        firm = os.path.basename(f).split("__")[1][:-4]
        s = pd.read_csv(f, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
        s["dprev"] = np.log(s.TRDPRC_1 / s.OPEN_PRC); s["relvol_prev"] = s.ACVOL_UNS / s.ACVOL_UNS.median()
        lag[firm] = s[["dprev", "relvol_prev"]]
    L = pd.concat([v.assign(firm=k) for k, v in lag.items()]).reset_index(names="prev")
    u0 = u0.merge(L, on=["firm", "prev"], how="left")
    u0["night"] = np.where(u0.typ == "W", "W", np.where(u0.prev.dt.dayofweek == 6, "Sun", "Mon-Wed"))
    rev = {}
    for k, df in u0.dropna(subset=["dprev"]).groupby("night"):
        x = df.dprev.values; y = df.g.values
        b = np.polyfit(x, y, 1)[0]
        rev[k] = dict(slope=float(b), corr=float(np.corrcoef(x, y)[0, 1]), n=int(len(df)))
    out["B3_reversal_gap_on_prev_intraday"] = rev
    # (iv) Sunday-night z2 by tercile of Sunday relative volume
    sn = u0[u0.night == "Sun"].dropna(subset=["relvol_prev"]).copy()
    sn["terc"] = pd.qcut(sn.relvol_prev.rank(method="first"), 3, labels=["low", "mid", "high"])
    baseN = u0[u0.typ == "N"].z2w.mean()
    out["B3_sunday_night_by_sunday_volume"] = (sn.groupby("terc", observed=True).z2w.mean() / baseN).round(3).to_dict()
    # (iii) denominator without Sunday sessions: rebuild s22 from intraday returns excluding Sundays
    dd = []
    for f in glob.glob("data/raw/lseg/job02/stocks/DFMGI__*.csv") + glob.glob("data/raw/lseg/job02/stocks/FTFADGI__*.csv"):
        firm = os.path.basename(f).split("__")[1][:-4]
        s = pd.read_csv(f, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
        d = np.log(s.TRDPRC_1 / s.OPEN_PRC); d[s.index.dayofweek == 6] = np.nan
        s22ns = (d ** 2).shift(1).rolling(30, min_periods=15).mean()
        dd.append(pd.DataFrame(dict(firm=firm, date=s.index, s22ns=s22ns.values)))
    D = pd.concat(dd)
    u1 = mk["UAE"].merge(D, on=["firm", "date"], how="left")
    u1["z2w"] = np.minimum(u1.g ** 2 / u1.s22ns, (u1.g ** 2 / u1.s22ns).quantile(0.999))
    out["B3_UAE_denominator_without_Sundays"] = summary(mt.market_tests(u1.dropna(subset=["z2w"])))
    # ---- B4 66-session denominators
    b4 = {}
    for k, v in mk.items():
        vv = []
        for (m, firm), df in g[g.market.isin(["DFMGI", "FTFADGI"] if k == "UAE" else ["TASI"])].groupby(["market", "firm"]):
            df = df.sort_values("date").copy()
            df["s66"] = (df.d ** 2).shift(1).rolling(66, min_periods=45).mean()
            vv.append(df)
        V = pd.concat(vv); V = V[V.inwin == 1].dropna(subset=["s66"])
        z = V.g ** 2 / V.s66; V["z2w"] = np.minimum(z, z.quantile(0.999))
        b4[k] = summary(mt.market_tests(V))
    out["B4_s66"] = b4
    # ---- B5 Saudi price limits
    k5 = mk["KSA"][mk["KSA"].g.abs() < 0.095]
    out["B5_KSA_no_limit_gaps"] = summary(mt.market_tests(k5))
    out["B5_KSA_limit_share"] = (mk["KSA"].assign(lim=mk["KSA"].g.abs() >= 0.095).groupby(["post", "typ"]).lim.mean()).round(4).to_dict()
    out["B5_KSA_limit_share"] = {f"{a}|{b}": v for (a, b), v in out["B5_KSA_limit_share"].items()}
    # ---- B6 Ramadan excluded
    out["B6_no_ramadan"] = {k: summary(mt.market_tests(v[~in_ramadan(v.date)])) for k, v in mk.items()}
    json.dump(out, open("output/results_gcc_exploratory.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:6000])

if __name__ == "__main__":
    main()
