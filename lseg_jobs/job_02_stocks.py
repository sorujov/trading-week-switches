# job_02_stocks: constituent OHLCV for DFM, ADX, Tadawul (selection rule in pap/02_pap_GCC.md s.3) + SPX dates
import os, pandas as pd, numpy as np, traceback, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
from dotenv import load_dotenv; load_dotenv(r".env")
import lseg.data as ld
OUT = "data/job02"; os.makedirs(OUT + "/stocks", exist_ok=True)
ld.open_session(app_key=os.environ["LSEG_APP_KEY"])
F = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"]
try:
    plan = {"TASI": ("0#.TASI", "2017-06-30", "2008-06-01", "2018-07-31"),
            "DFMGI": ("0#.DFMGI", "2026-09-25", "2017-06-01", "2026-09-26"),
            "FTFADGI": ("0#.FTFADGI", "2026-09-25", "2017-06-01", "2026-09-26")}
    for mkt, (chain, mcdate, s, e) in plan.items():
        try:
            u = ld.get_data(chain, ["TR.CommonName"])
            rics = u["Instrument"].dropna().tolist()
            mc = ld.get_data(rics, ["TR.CompanyMarketCap", "TR.CompanyMarketCap.date"], parameters={"SDate": mcdate, "Curn": "USD"})
            mc.to_csv(f"{OUT}/mcap_{mkt}.csv", index=False)
            col = [c for c in mc.columns if "Market Cap" in c and "Date" not in c][0]
            mc[col] = pd.to_numeric(mc[col], errors="coerce")
            top = mc.sort_values(col, ascending=False).dropna(subset=[col]).head(45)
            print(mkt, "universe", len(rics), "with mcap", mc[col].notna().sum()); print(top.head(5).to_string())
            for ric in top["Instrument"]:
                try:
                    d = ld.get_history(ric, fields=F, interval="daily", start=s, end=e)
                    d.to_csv(f"{OUT}/stocks/{mkt}__{ric}.csv")
                    print(f"  {ric}: {len(d)} rows {d.index.min().date()}..{d.index.max().date()}")
                except Exception as ex: print("  ERROR", ric, type(ex).__name__, str(ex)[:150])
        except Exception as ex:
            print("ERROR market", mkt, type(ex).__name__, str(ex)[:300]); traceback.print_exc()
    try:
        d = ld.get_history(".SPX", fields=["OPEN_PRC", "TRDPRC_1"], interval="daily", start="2005-01-01", end="2026-09-26")
        d.to_csv(f"{OUT}/SPX.csv"); print("SPX", d.shape, d.index.min(), d.index.max())
    except Exception as ex: print("ERROR SPX", ex)
finally:
    ld.close_session()
print("RESULT: DONE")
