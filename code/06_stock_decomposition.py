"""Exploratory (post-registration): Sunday-night contrast for dual-listed vs domestic stocks, cell medians,
and a three-component decomposition fitted on pre-switch cells and used to predict the post-switch weekend ratio."""
import pandas as pd, numpy as np, json, statsmodels.formula.api as smf
rng = np.random.default_rng(11)
st = pd.read_csv("data/clean/stock_gaps.csv", parse_dates=["date"], dtype={"sec": str})
out = {}
# 1) Sunday-night contrast, pre period, N and S gaps
pre = st[(st.post == 0) & st.typ.isin(["N", "S"])].assign(S=lambda x: (x.typ == "S").astype(int))
cl = pd.factorize(pre.week)[0]
m1 = smf.ols("z2 ~ S*DL", pre).fit(cov_type="cluster", cov_kwds={"groups": cl})
pz = pre[pre.z2 > 1e-10].assign(lz=lambda x: 0.5 * np.log(x.z2))
m2 = smf.ols("lz ~ S*DL + C(sec)", pz).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pz.week)[0]})
out["sunday_contrast_levels"] = {k: dict(coef=m1.params[k], se=m1.bse[k], p=m1.pvalues[k]) for k in m1.params.index}
out["sunday_contrast_logabs"] = {k: dict(coef=m2.params[k], se=m2.bse[k], p=m2.pvalues[k]) for k in m2.params.index if not k.startswith("C(")}
# 2) cell table: mean z2, median |z|, ratios to N within group-period
rows = []
for (dl, p, t), x in st.groupby(["DL", "post", "typ"]):
    rows.append(dict(DL=dl, post=p, typ=t, n=len(x), stocks=x.sec.nunique(), mean_z2=x.z2.mean(), med_absz=np.median(np.sqrt(x.z2)),
                     H=x.H.mean(), U=x.U.mean(), Ufut=x.Ufut.mean()))
cells = pd.DataFrame(rows)
for (dl, p), x in cells.groupby(["DL", "post"]):
    nrow = x[x.typ == "N"].iloc[0]
    cells.loc[x.index, "R_mean"] = x.mean_z2 / nrow.mean_z2
    cells.loc[x.index, "R_median"] = (x.med_absz / nrow.med_absz) ** 2
cells.to_csv("output/tables/stock_cells.csv", index=False)
out["cells"] = cells.to_dict(orient="records")
# 3) decomposition for DL stocks: ratio(gap) = a + b*U + c*Weekend, exactly identified on pre cells S, N, W;
#    predict post weekend ratio; block-bootstrap over weeks (jointly) for an interval on (observed - predicted).
def fit_predict(df):
    c = df.groupby(["post", "typ"]).z2.mean()
    Ucell = df.groupby(["post", "typ"]).U.mean()
    N, S, W = c[(0, "N")], c[(0, "S")], c[(0, "W")]
    rS, rW = S / N, W / N
    uN, uW, uWp = Ucell[(0, "N")], Ucell[(0, "W")], Ucell[(1, "W")]
    b = (1 - rS) / uN           # per-session increment (relative to N)
    a = rS                       # no-session baseline
    cw = rW - a - b * uW         # weekend (calendar) component
    pred = a + b * uWp + cw
    obs = c[(1, "W")] / c[(1, "N")]
    return dict(a=a, b=b, c_weekend=cw, pred_post=pred, obs_post=obs, pred_no_info=rW, pred_pure_info=a + b * uWp)
dl = st[(st.DL == 1) & st.typ.isin(["N", "S", "W"])]
base = fit_predict(dl)
weeks = {p: dl[dl.post == p].week.unique() for p in (0, 1)}
grp = {w: x for w, x in dl.groupby("week")}
bs = []
for bb in range(999):
    parts = []
    for p in (0, 1):
        ws = weeks[p]; n = len(ws); k = rng.integers(n); idx = []
        for j in range(n):
            if j > 0 and rng.random() < 0.25: k = rng.integers(n)
            idx.append(ws[k]); k = (k + 1) % n
        parts += [grp[w] for w in idx]
    r = fit_predict(pd.concat(parts)); bs.append([r["obs_post"] - r["pred_post"], r["obs_post"] - r["pred_no_info"], r["obs_post"] - r["pred_pure_info"], r["b"], r["c_weekend"]])
bs = np.array(bs)
base["ci_obs_minus_pred"] = list(np.percentile(bs[:, 0], [2.5, 97.5]))
base["ci_obs_minus_noinfo"] = list(np.percentile(bs[:, 1], [2.5, 97.5]))
base["ci_obs_minus_pureinfo"] = list(np.percentile(bs[:, 2], [2.5, 97.5]))
base["ci_b"] = list(np.percentile(bs[:, 3], [2.5, 97.5])); base["ci_c"] = list(np.percentile(bs[:, 4], [2.5, 97.5]))
out["DL_decomposition"] = base
json.dump(out, open("output/results_stock_decomposition.json", "w"), indent=1, default=float)
print(cells.round(3).to_string())
print(json.dumps({k: out[k] for k in ["sunday_contrast_levels", "sunday_contrast_logabs", "DL_decomposition"]}, indent=1, default=float))
