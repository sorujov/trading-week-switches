"""Build stock-level gap panels for the Gulf switches (plan: pap/02_pap_GCC.md).

Inputs  data/raw/lseg/job01/{DFMGI,FTFADGI,TASI}_rt.csv   exchange session calendars (index dates)
        data/raw/lseg/job02/stocks/<MKT>__<RIC>.csv        daily OPEN_PRC, HIGH_1, LOW_1, TRDPRC_1, ACVOL_UNS
        data/raw/lseg/job02/mcap_<MKT>.csv                 market caps on the selection date
        data/raw/tase/GSPC.csv                             NYSE session calendar
Output  data/clean/gcc_gaps.csv, output/gcc_selection.json, output/gcc_calendar_summary.csv
Run from the project root.
"""
import pandas as pd, numpy as np, glob, os, json, sys
from datetime import datetime, time
from zoneinfo import ZoneInfo
sys.path.insert(0, "code")
NY = ZoneInfo("America/New_York")
NYSE_HALF = eval(open("code/01_build.py").read().split("NYSE_HALF = ")[1].split("\n\n")[0])

MKT = {  # tz, switch date (first session of new week), weekend days before/after (Mon=0), hours before/after, window
    "DFMGI":   dict(tz="Asia/Dubai", switch="2022-01-03", wk0={4, 5}, wk1={5, 6},
                    hrs0=((10, 0), (14, 0)), hrs1=((10, 0), (15, 0)), win=("2018-01-02", "2026-09-25"), sel="2026-09-25"),
    "FTFADGI": dict(tz="Asia/Dubai", switch="2022-01-03", wk0={4, 5}, wk1={5, 6},
                    hrs0=((10, 0), (14, 0)), hrs1=((10, 0), (15, 0)), win=("2018-01-02", "2026-09-25"), sel="2026-09-25"),
    "TASI":    dict(tz="Asia/Riyadh", switch="2013-06-30", wk0={3, 4}, wk1={4, 5},
                    hrs0=((11, 0), (15, 30)), hrs1=((11, 0), (15, 30)), win=("2009-07-01", "2017-06-30"), sel="2017-06-30"),
}
N_SEL = 30

def nyse():
    spx = pd.read_csv("data/raw/tase/GSPC.csv", index_col=0, parse_dates=True).sort_index()
    idx = spx.index
    if os.path.exists("data/raw/lseg/job02/SPX.csv"):   # extend the calendar past the Yahoo file's end
        idx = idx.union(pd.read_csv("data/raw/lseg/job02/SPX.csv", index_col=0, parse_dates=True).index.normalize())
    # the Yahoo file ends 23 Sep 2026; 24 and 25 Sep 2026 were regular NYSE sessions (no holiday, no early close)
    idx = idx.union(pd.DatetimeIndex(["2026-09-24", "2026-09-25"]))
    spx = pd.DataFrame(index=idx[idx >= "2007-01-01"].unique())
    o = np.array([datetime.combine(d.date(), time(9, 30), NY).timestamp() for d in spx.index])
    c = np.array([datetime.combine(d.date(), time(13, 0) if d in NYSE_HALF else time(16, 0), NY).timestamp() for d in spx.index])
    return o, c

def week_id(dates):
    d = pd.to_datetime(dates) + pd.Timedelta(days=2)
    iso = d.dt.isocalendar()
    return iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

def calendar(m, cfg, so, sc):
    tz = ZoneInfo(cfg["tz"])
    cal = pd.read_csv(f"data/raw/lseg/job01/{m}_rt.csv", index_col=0, parse_dates=True).index.normalize().unique().sort_values()
    sw = pd.Timestamp(cfg["switch"])
    rows = []
    for d0, d1 in zip(cal[:-1], cal[1:]):
        post0, post1 = d0 >= sw, d1 >= sw
        h0 = cfg["hrs1"] if post0 else cfg["hrs0"]; h1 = cfg["hrs1"] if post1 else cfg["hrs0"]
        a = datetime.combine(d0.date(), time(*h0[1]), tz); b = datetime.combine(d1.date(), time(*h1[0]), tz)
        H = (b - a).total_seconds() / 3600
        ov = np.clip(np.minimum(sc, b.timestamp()) - np.maximum(so, a.timestamp()), 0, None)
        U = ov.sum() / (6.5 * 3600)
        mid = pd.date_range(d0 + pd.Timedelta(days=1), d1 - pd.Timedelta(days=1))
        if post0 != post1:
            typ = "X"  # transition closure
        else:
            wk = cfg["wk1"] if post1 else cfg["wk0"]
            if len(mid) == 0:
                typ = "N" if U > 0.999 else ("Z" if U < 1e-9 else "O")
            elif all(x.dayofweek in wk for x in mid):
                typ = "W"
            else:
                typ = "H"
        rows.append(dict(date=d1, prev=d0, post=int(post1), typ=typ, H=H, U=U))
    c = pd.DataFrame(rows)
    # calendar-week id common to all markets: Saturday-to-Friday blocks (a session on date d belongs to the ISO week of d + 2 days),
    # so that every market's sessions of one week, and the first session after its weekend, share a cluster
    c["week"] = week_id(c.date)
    return c

def load_stock(f):
    s = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
    s.index = s.index.normalize(); s = s[~s.index.duplicated()]
    s = s.rename(columns={"OPEN_PRC": "open", "HIGH_1": "high", "LOW_1": "low", "TRDPRC_1": "close", "ACVOL_UNS": "vol"})
    return s.apply(pd.to_numeric, errors="coerce")

def main():
    so, sc = nyse()
    out, sel_info, calsum = [], {}, []
    for m, cfg in MKT.items():
        c = calendar(m, cfg, so, sc)
        w0, w1 = pd.Timestamp(cfg["win"][0]), pd.Timestamp(cfg["win"][1])
        cw = c[(c.date >= w0) & (c.date <= w1)]
        calsum.append(cw.groupby(["post", "typ"]).agg(n=("H", "size"), H=("H", "mean"), U=("U", "mean")).reset_index().assign(market=m))
        # --- selection rule (plan s.3): 30 largest by market cap on the selection date, full coverage of the window
        mc = pd.read_csv(f"data/raw/lseg/job02/mcap_{m}.csv")
        col = [k for k in mc.columns if "Market Cap" in k and "Date" not in k][0]
        mc[col] = pd.to_numeric(mc[col], errors="coerce")
        mc = mc.dropna(subset=[col]).sort_values(col, ascending=False)
        chosen, skipped = [], []
        for ric in mc.Instrument:
            f = f"data/raw/lseg/job02/stocks/{m}__{ric}.csv"
            if not os.path.exists(f): skipped.append((ric, "no file")); continue
            s = load_stock(f).dropna(subset=["close"])
            if len(s) == 0 or s.index.min() > w0 + pd.Timedelta(days=10) or s.index.max() < w1 - pd.Timedelta(days=10):
                skipped.append((ric, "coverage")); continue
            chosen.append(ric)
            if len(chosen) == N_SEL: break
        sel_info[m] = dict(chosen=chosen, skipped_before_30=skipped)
        for ric in chosen:
            s = load_stock(f"data/raw/lseg/job02/stocks/{m}__{ric}.csv")
            s["d"] = np.log(s.close / s.open)
            s.loc[~np.isfinite(s.d), "d"] = np.nan
            s["s22"] = (s.d ** 2).shift(1).rolling(22, min_periods=15).mean()
            s["prev_date"] = s.index.to_series().shift(1)
            s["prev_close"] = s.close.shift(1)
            s = s.reset_index(names="date").merge(c, on="date", how="inner")
            s = s[s.prev_date == s.prev]                      # previous stock row is the previous exchange session
            s["g"] = np.log(s.open / s.prev_close)
            v5 = s.vol.quantile(0.05)
            bad = s.g.isna() | ~np.isfinite(s.g) | (s.vol.fillna(0) <= 0) | ((s.open == s.prev_close) & (s.vol < v5))
            s = s[~bad & (s.s22 > 0)]
            s["inwin"] = ((s.date >= w0) & (s.date <= w1)).astype(int)
            s["z2"] = s.g ** 2 / s.s22
            s["market"], s["firm"] = m, ric
            out.append(s[["market", "firm", "date", "prev", "post", "typ", "H", "U", "week", "g", "d", "s22", "z2", "vol", "inwin"]])
    g = pd.concat(out, ignore_index=True)
    # winsorise z2 at the 99.9th percentile of the market's pooled z2 (plan s.3)
    qs = g[g.inwin == 1].groupby("market").z2.quantile(0.999)
    q = g.market.map(qs)
    g["z2w"] = np.minimum(g.z2, q)
    os.makedirs("data/clean", exist_ok=True); os.makedirs("output", exist_ok=True)
    g.to_csv("data/clean/gcc_gaps.csv", index=False)
    json.dump(sel_info, open("output/gcc_selection.json", "w"), indent=1)
    pd.concat(calsum).to_csv("output/gcc_calendar_summary.csv", index=False)
    print(pd.concat(calsum).round(2).to_string())
    print(g.groupby(["market", "post", "typ"]).agg(n=("z2", "size"), firms=("firm", "nunique")).to_string())
    print({k: len(v["chosen"]) for k, v in sel_info.items()})

if __name__ == "__main__":
    main()
