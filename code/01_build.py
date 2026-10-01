"""Build the gap-level dataset.

Inputs
  data/raw/tase/tase_TA125_official.csv, data/raw/tase/tase_TA35_official.csv  (official TASE EOD)
  data/raw/tase/GSPC.csv                                                   (Yahoo ^GSPC, NYSE session calendar)
Output
  data/clean/gaps_<INDEX>.csv
"""
import pandas as pd, numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

IL, NY = ZoneInfo("Asia/Jerusalem"), ZoneInfo("America/New_York")
SWITCH = pd.Timestamp("2026-01-05")
SUNDAY_SHORT = pd.Timestamp("2020-03-08")   # Sunday close moved to ~15:50 (WFE Focus, TASE notice)

def tase_close_time(d):
    wd = d.dayofweek  # Mon=0 ... Sun=6
    if wd == 6:  # Sunday (pre-switch only)
        return time(15, 50) if d >= SUNDAY_SHORT else time(16, 25)
    if wd == 4:  # Friday (post-switch only)
        return time(13, 45)
    return time(17, 25)

def tase_open_time(d):
    return time(10, 0)

# NYSE early closes (13:00 ET), from NYSE holiday calendars 2005-2026
NYSE_HALF = set(pd.to_datetime([
    "2005-11-25","2006-07-03","2006-11-24","2007-07-03","2007-11-23","2007-12-24","2008-07-03","2008-11-28","2008-12-24",
    "2009-11-27","2009-12-24","2010-11-26","2011-11-25","2012-07-03","2012-11-23","2012-12-24","2013-07-03","2013-11-29",
    "2013-12-24","2014-07-03","2014-11-28","2014-12-24","2015-11-27","2015-12-24","2016-11-25","2017-07-03","2017-11-24",
    "2018-07-03","2018-11-23","2018-12-24","2019-07-03","2019-11-29","2019-12-24","2020-11-27","2020-12-24","2021-11-26",
    "2022-11-25","2023-07-03","2023-11-24","2024-07-03","2024-11-29","2024-12-24","2025-07-03","2025-11-28","2025-12-24",
    "2026-07-02"]))

def build_nyse_sessions(spx):
    sess = []
    for d in spx.index:
        o = datetime.combine(d.date(), time(9, 30), NY)
        c = datetime.combine(d.date(), time(13, 0) if d.normalize() in NYSE_HALF else time(16, 0), NY)
        sess.append((d, o, c))
    return sess

def main(name):
    t = pd.read_csv(f"data/raw/tase/tase_{name}_official.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    spx = pd.read_csv("data/raw/tase/GSPC.csv", index_col=0, parse_dates=True).sort_index()
    spx = spx[spx.index >= "2004-12-01"]
    spx["r"] = np.log(spx["Close"]).diff()
    sess = build_nyse_sessions(spx)
    s_open = np.array([s[1].timestamp() for s in sess]); s_close = np.array([s[2].timestamp() for s in sess])

    t["g"] = np.log(t["open"] / t["close"].shift(1))
    t["d"] = np.log(t["close"] / t["open"])
    t["cc"] = np.log(t["close"] / t["close"].shift(1))
    # consistency check: TASE 'base' price equals previous close (except index rebasings)
    t["base_chk"] = (t["base"] / t["close"].shift(1) - 1).abs()

    rows = []
    for i in range(1, len(t)):
        d0, d1 = t.loc[i - 1, "date"], t.loc[i, "date"]
        start = datetime.combine(d0.date(), tase_close_time(d0), IL)
        end = datetime.combine(d1.date(), tase_open_time(d1), IL)
        H = (end - start).total_seconds() / 3600
        a, b = start.timestamp(), end.timestamp()
        ov = np.clip(np.minimum(s_close, b) - np.maximum(s_open, a), 0, None)
        U = ov.sum() / (6.5 * 3600)
        # US sessions whose close falls inside the gap
        idx = np.where((s_close > a) & (s_close <= b))[0]
        us_r = [spx["r"].iloc[k] for k in idx]
        us_wd = [sess[k][0].dayofweek for k in idx]
        post = d1 >= SWITCH
        cal = (d1 - d0).days
        w0, w1 = d0.dayofweek, d1.dayofweek
        if not post and d0 < SWITCH:
            if w0 == 3 and w1 == 6 and cal == 3: typ = "W"
            elif w0 == 6 and w1 == 0 and cal == 1: typ = "S"
            elif cal == 1 and w0 in (0, 1, 2): typ = "N"
            else: typ = "H"
        elif post and d0 >= SWITCH:
            if w0 == 4 and w1 == 0 and cal == 3: typ = "W"
            elif cal == 1 and w0 in (0, 1, 2, 3): typ = "N"
            else: typ = "H"
        else:
            typ = "H"  # transition gap
        rows.append(dict(date=d1, prev=d0, typ=typ, post=int(post), H=H, U=U,
                         n_us=len(idx), us_sum=float(np.sum(us_r)) if us_r else 0.0,
                         us_thu=float(sum(r for r, w in zip(us_r, us_wd) if w == 3)),
                         us_fri=float(sum(r for r, w in zip(us_r, us_wd) if w == 4)),
                         us_other=float(sum(r for r, w in zip(us_r, us_wd) if w not in (3, 4))),
                         g=t.loc[i, "g"], d=t.loc[i, "d"], cc=t.loc[i, "cc"], base_chk=t.loc[i, "base_chk"]))
    g = pd.DataFrame(rows)
    # local volatility from intraday returns of the 22 (and 66) previous sessions
    d2 = t["d"] ** 2
    t["s22"] = d2.shift(1).rolling(22, min_periods=15).mean()
    t["s66"] = d2.shift(1).rolling(66, min_periods=45).mean()
    g = g.merge(t[["date", "s22", "s66"]], on="date", how="left")
    g["z2"] = g["g"] ** 2 / g["s22"]
    g["z2_66"] = g["g"] ** 2 / g["s66"]
    g.to_csv(f"data/clean/gaps_{name}.csv", index=False)
    print(name, "rows", len(g))
    print(g.groupby(["post", "typ"]).agg(n=("g", "size"), H=("H", "mean"), U=("U", "mean"), n_us=("n_us", "mean")).round(2))
    print("base != prev close (>0.1%):", int((g["base_chk"] > 1e-3).sum()))

if __name__ == "__main__":
    import os; os.makedirs("data/clean", exist_ok=True)
    for n in ["TA125", "TA35"]:
        main(n)
