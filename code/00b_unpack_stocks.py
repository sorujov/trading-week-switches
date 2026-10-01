"""Unpack gzipped/base64 dumps of TASE security EOD data (api.tase.co.il/api/security/historyeod, pType=8),
captured through the TASE website on 2026-09-23, into one CSV per security."""
import json, base64, gzip, io, sys, glob, pandas as pd
frames = []
for f in sys.argv[1:]:
    t = json.load(open(f))[0]["text"]
    t = t[t.index("LEN "):] if "LEN " in t else t
    t = t.split('"')[0] if t.startswith('"') else t
    b64 = t.split(" ", 2)[2].split('"')[0].split("\n")[0]
    raw = gzip.decompress(base64.b64decode(b64)).decode()
    frames.append(pd.read_csv(io.StringIO(raw), header=None, names=["sec", "date", "base", "open", "close", "high", "low", "turnover", "traded"], dtype={"sec": str}))
df = pd.concat(frames)
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
for sec, x in df.groupby("sec"):
    x.drop(columns="sec").drop_duplicates("date").sort_values("date").to_csv(f"data/raw/tase/stocks/{sec}.csv", index=False)
print(df.sec.nunique(), "securities,", len(df), "rows")
