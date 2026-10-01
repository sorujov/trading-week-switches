"""Assemble official TASE end-of-day index data captured from api.tase.co.il/api/index/historyeod
(pType=8, oId=137 TA-125, oId=142 TA-35), retrieved 2026-09-23 via the TASE website.

The CSV files this script writes are already in data/raw/tase/. To rebuild them, save the API
responses (JSON, one file per request) as data/raw/tase/api/<index>_<n>.txt, e.g. TA125_1.txt."""
import json, io, pandas as pd, re, sys
D='data/raw/tase/api/'
files={'TA125':['TA125_1','TA125_2','TA125_3'],'TA35':['TA35_1','TA35_2','TA35_3']}
for name,fs in files.items():
    txt=''
    for f in fs:
        t=json.load(open(D+f+'.txt'))[0]['text']
        t=t[:t.rfind('"')+1]
        txt+=json.loads(t).strip()+'\n'
    df=pd.read_csv(io.StringIO(txt),header=None,names=['date','base','open','close','high','low'])
    df['date']=pd.to_datetime(df['date'],format='%d/%m/%Y')
    n0=len(df); df=df.drop_duplicates('date').sort_values('date')
    df.to_csv(f'data/raw/tase/tase_{name}_official.csv',index=False)
    print(name,n0,len(df),df.date.min().date(),df.date.max().date())
    print(pd.crosstab(df.date.dt.year,df.date.dt.day_name()).loc[[2005,2013,2014,2025,2026]].to_string())
