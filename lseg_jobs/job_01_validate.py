# job_01_validate: index OHLC for switching markets + chain constituents. Prints summaries only.
import os, pandas as pd, numpy as np, traceback
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
from dotenv import load_dotenv; load_dotenv(r".env")
import lseg.data as ld
OUT = "data/job01"; os.makedirs(OUT, exist_ok=True)
ld.open_session(app_key=os.environ["LSEG_APP_KEY"])
def summ(name, df):
    df = df.copy(); df.index = pd.to_datetime(df.index)
    print(f"--- {name}: shape {df.shape}, {df.index.min().date() if len(df) else None} .. {df.index.max().date() if len(df) else None}")
    print("columns:", list(df.columns))
    if len(df):
        wd = pd.crosstab(df.index.year, df.index.day_name().str[:3])
        print(wd.to_string())
try:
    for q in ["Tadawul All Share index", "DFM General Index", "ADX General Index", "FTSE ADX General Index", "TA-125"]:
        try:
            r = ld.discovery.search(query=q, view=ld.discovery.Views.INDEX_INSTRUMENTS, top=5, select="RIC,DTSubjectName")
            print("SEARCH", q); print(r.to_string())
        except Exception as e: print("ERROR search", q, type(e).__name__, e)
    rics = [".TA125", ".TASI", ".DFMGI", ".FTFADGI", ".ADI", ".MSM30", ".KWSEIDX", ".BKA"]
    for ric in rics:
        for tag, kw in [("rt", dict(fields=["OPEN_PRC","HIGH_1","LOW_1","TRDPRC_1"], interval="daily")),
                        ("tr", dict(fields=["TR.OpenPrice","TR.HighPrice","TR.LowPrice","TR.PriceClose"]))]:
            try:
                df = ld.get_history(ric, start="2005-01-01", end="2026-09-26", **kw)
                df.to_csv(f"{OUT}/{ric.strip('.')}_{tag}.csv"); summ(f"{ric} {tag}", df)
                c = df.columns
                o = pd.to_numeric(df[c[0]], errors="coerce"); cl = pd.to_numeric(df[c[-1]], errors="coerce")
                stale = (np.abs(o / cl.shift(1) - 1) < 1e-6).groupby(df.index.year).mean()
                print("share open==prev close by year:", stale.round(3).to_dict())
            except Exception as e: print("ERROR", ric, tag, type(e).__name__, str(e)[:200])
    for chain in ["0#.TASI", "0#.DFMGI", "0#.FTFADGI", "0#.TA125"]:
        try:
            d = ld.get_data(chain, ["TR.CommonName", "TR.CompanyMarketCap"])
            d.to_csv(f"{OUT}/chain_{chain.strip('0#.')}.csv"); print("CHAIN", chain, d.shape); print(d.head(3).to_string())
        except Exception as e: print("ERROR chain", chain, type(e).__name__, str(e)[:200])
except Exception:
    traceback.print_exc()
finally:
    ld.close_session()
print("RESULT: DONE")
