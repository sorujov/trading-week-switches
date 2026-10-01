"""P5 (vendor exhibit) and Addendum A2/A3 (exploratory, post-registration) at the index level."""
import pandas as pd, numpy as np, json, statsmodels.formula.api as smf
from datetime import datetime, time
from zoneinfo import ZoneInfo
IL, NY = ZoneInfo("Asia/Jerusalem"), ZoneInfo("America/New_York")
out = {}

# ---------- P5: Yahoo vs official ----------
off = pd.read_csv("data/raw/tase/tase_TA125_official.csv", parse_dates=["date"]).set_index("date")
yh = pd.read_csv("data/raw/tase/TA125.TA.csv", index_col=0, parse_dates=True)
yh.index = pd.to_datetime(yh.index).tz_localize(None).normalize()
# match each Yahoo row to the official session with the same close (within 0.01%)
m = []
offc = off["close"]
for d, r in yh.iterrows():
    cand = offc[(offc.index >= d - pd.Timedelta(days=3)) & (offc.index <= d + pd.Timedelta(days=3))]
    hit = cand[(cand / r["Close"] - 1).abs() < 1e-4]
    m.append(dict(ydate=d, odate=hit.index[0] if len(hit) else pd.NaT))
m = pd.DataFrame(m)
m["lag"] = (m.ydate - m.odate).dt.days
m["year"] = m.ydate.dt.year
lagtab = m.groupby("year").agg(n_yahoo=("ydate", "size"), matched=("odate", lambda s: s.notna().sum()),
                               lag0=("lag", lambda s: (s == 0).sum()), lag1=("lag", lambda s: (s == 1).sum()))
lagtab["n_official"] = off.groupby(off.index.year).size().reindex(lagtab.index)
lagtab["omitted"] = lagtab["n_official"] - lagtab["matched"]
omitted_dow = off.index.difference(pd.DatetimeIndex(m.odate.dropna())).day_name().value_counts()
out["P5_lagtab"] = lagtab.reset_index().to_dict(orient="records")
out["P5_omitted_by_weekday_official"] = omitted_dow.to_dict()
lagtab.to_csv("output/tables/vendor_lag_by_year.csv")

def dow_reg(s, label, a="2014-01-01", b="2025-12-31"):
    s = s[(s.index >= a) & (s.index <= b)]
    r = np.log(s).diff().dropna() * 100
    df = pd.DataFrame({"r": r, "dow": r.index.day_name()})
    base = "Tuesday"
    mm = smf.ols(f"r ~ C(dow, Treatment('{base}'))", df).fit(cov_type="HC1")
    var = df.groupby("dow").r.agg(["size", "mean", "var"])
    return dict(label=label, coef=mm.params.to_dict(), p=mm.pvalues.to_dict(), var=var.to_dict(orient="index"))
out["P5_dow_official"] = dow_reg(off["close"], "official")
out["P5_dow_yahoo"] = dow_reg(yh["Close"], "yahoo")

# ---------- A2: futures clock ----------
spx = pd.read_csv("data/raw/tase/GSPC.csv", index_col=0, parse_dates=True)
def fut_hours(a, b):
    """CME Globex equity index futures: open Sun 18:00 ET to Fri 17:00 ET, daily halt 17:00-18:00 ET.
    Holidays ignored (approximation)."""
    t = pd.date_range(a, b, freq="15min", inclusive="left")
    tn = t.tz_convert(NY)
    wd, hr = tn.dayofweek, tn.hour
    openm = ((wd <= 3) & (hr != 17)) | ((wd == 4) & (hr < 17)) | ((wd == 6) & (hr >= 18))
    return openm.sum() * 0.25
g = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date", "prev"])
def tase_close(d):
    wd = d.dayofweek
    if wd == 6: return time(15, 50) if d >= pd.Timestamp("2020-03-08") else time(16, 25)
    if wd == 4: return time(13, 45)
    return time(17, 25)
Uf = []
for _, r in g.iterrows():
    a = pd.Timestamp(datetime.combine(r.prev.date(), tase_close(r.prev), IL))
    b = pd.Timestamp(datetime.combine(r.date.date(), time(10, 0), IL))
    Uf.append(fut_hours(a, b))
g["Ufut"] = Uf
g.to_csv("data/clean/gaps_TA125.csv", index=False)
sub = g[(g.date >= "2020-03-08") & g.typ.isin(["N", "W", "S"])].dropna(subset=["z2"])
cells = sub.groupby(["post", "typ"]).agg(n=("z2", "size"), H=("H", "mean"), U=("U", "mean"), Ufut=("Ufut", "mean"), mean_z2=("z2", "mean"), median_absz=("z2", lambda x: np.median(np.sqrt(x))))
out["A2_cells"] = cells.reset_index().to_dict(orient="records")
cells.to_csv("output/tables/gap_cells.csv")
res = {}
for f in ["z2 ~ H + U", "z2 ~ H + Ufut", "z2 ~ H", "z2 ~ Ufut", "z2 ~ U", "z2 ~ H + U + Ufut"]:
    mm = smf.ols(f, sub).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    res[f] = dict(coef=mm.params.to_dict(), se=mm.bse.to_dict(), p=mm.pvalues.to_dict(), aic=mm.aic, r2=mm.rsquared, n=int(mm.nobs))
out["A2_models"] = res

# ---------- A3: robust scale, log|z| ----------
sub = sub.assign(lz=np.log(np.sqrt(sub.z2) + 1e-8), W=(sub.typ == "W").astype(int), S=(sub.typ == "S").astype(int))
sub = sub[np.sqrt(sub.z2) > 1e-6]   # drop exact zero gaps
mm = smf.ols("lz ~ W + post + W:post + S", sub).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
out["A3_logabs"] = dict(coef=mm.params.to_dict(), se=mm.bse.to_dict(), p=mm.pvalues.to_dict(), n=int(mm.nobs),
                        note="2*coef is the log variance-scale difference under a location-scale model")
json.dump(out, open("output/results_vendor_exploratory.json", "w"), indent=1, default=str)
print(lagtab.to_string()); print(omitted_dow)
for k in ["P5_dow_official", "P5_dow_yahoo"]:
    print(k, {a: round(b, 3) for a, b in out[k]["coef"].items()}, {a: round(b, 3) for a, b in out[k]["p"].items()})
    print(pd.DataFrame(out[k]["var"]).T.round(3))
print(cells.round(3).to_string())
for f, v in res.items(): print(f, {k: round(x, 4) for k, x in v["coef"].items()}, {k: round(x, 4) for k, x in v["p"].items()}, "AIC", round(v["aic"], 1))
print("A3", {k: round(v, 4) for k, v in out["A3_logabs"]["coef"].items()}, {k: round(v, 4) for k, v in out["A3_logabs"]["p"].items()})
