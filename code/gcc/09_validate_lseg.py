"""Step 1 of the plan: validate LSEG session calendars and index values.
Input  data/raw/lseg/job01/*_rt.csv, data/raw/tase/tase_TA125_official.csv
Output output/lseg_validation.json
"""
import pandas as pd, numpy as np, json
def rd(f):
    d = pd.read_csv(f, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce"); d.columns = ["o", "h", "l", "c"]; return d
out = {}
off = pd.read_csv("data/raw/tase/tase_TA125_official.csv", parse_dates=["date"]).set_index("date")
L = rd("data/raw/lseg/job01/TA125_rt.csv"); L = L[L.index <= off.index.max()]
j = off.join(L, how="inner")
out["TA125"] = dict(official_sessions=int(len(off)), lseg_sessions=int(len(L)),
                    official_not_in_lseg=int(len(off.index.difference(L.index))), lseg_not_in_official=int(len(L.index.difference(off.index))),
                    close_diff_gt_1e4=int((np.abs(j.c / j.close - 1) > 1e-4).sum()), compared=int(len(j)),
                    open_diff_gt_1e4_by_year=(np.abs(j.o / j.open - 1) > 1e-4).groupby(j.index.year).mean().round(3).to_dict())
for m in ("TA125", "TASI", "DFMGI", "FTFADGI"):
    d = rd(f"data/raw/lseg/job01/{m}_rt.csv")
    stale = (np.abs(d.o / d.c.shift(1) - 1) < 1e-6)
    out.setdefault("stale_open_share_by_year", {})[m] = stale.groupby(d.index.year).mean().round(3).to_dict()
json.dump(out, open("output/lseg_validation.json", "w"), indent=1, default=str)
print(json.dumps(out["TA125"], default=str))
