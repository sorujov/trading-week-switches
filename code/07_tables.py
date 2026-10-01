"""Write LaTeX tables from the JSON outputs (no numbers typed by hand)."""
import json, pandas as pd, numpy as np
R = json.load(open("output/results.json"))
E = json.load(open("output/results_vendor_exploratory.json"))
S = json.load(open("output/results_stocks.json"))
D = json.load(open("output/results_stock_decomposition.json"))
T = "output/tables/"
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
def star(p): return "^{***}" if p < .01 else "^{**}" if p < .05 else "^{*}" if p < .10 else ""

# Table 1: gap cells, index
cells = pd.DataFrame(E["A2_cells"])
name = {("0", "N"): "Weeknight, before", ("0", "S"): "Sunday night, before", ("0", "W"): "Weekend, before",
        ("1", "N"): "Weeknight, after", ("1", "W"): "Weekend, after"}
lines = []
for _, r in cells.iterrows():
    k = (str(int(r.post)), r.typ)
    nN = cells[(cells.post == r.post) & (cells.typ == "N")].iloc[0]
    lines.append(f"{name[k]} & {int(r.n)} & {r.H:.1f} & {r.U:.2f} & {r.Ufut:.1f} & {r.mean_z2:.3f} & {r.mean_z2 / nN.mean_z2:.2f} & {(r.median_absz / nN.median_absz) ** 2:.2f} \\\\")
open(T + "tab1_cells.tex", "w").write("\n".join(lines))

# Table 2: P1 primary and robustness
def p1row(lab, r):
    return f"{lab} & {f2(r['R_pre'])} & {f2(r['R_post'])} & {r['logdiff']:.2f} & ({r['logdiff_se']:.2f}) & {r['p_boot']:.2f} & {r['reg']['coef']['WxPost']:.2f} & ({r['reg']['se']['WxPost']:.2f}) & {r['n_W_post']} \\\\"
rows = [p1row("Primary (TA-125, $z^2$)", R["P1"]), p1row("Levels $g^2$", R["P1_levels"])]
lab = {"TA35": "TA-35", "s66": "66-session normaliser", "excl2020": "Excluding 2020", "excl_war23": "Excl. Oct--Dec 2023",
       "excl_jun25": "Excluding June 2025", "long_pre": "Pre sample from 2005", "winsor": "Winsorised 1/99",
       "excl_hol_adj": "Excl. holiday-adjacent", "placebo_2025": "Placebo switch, 5 Jan 2025"}
rows += [p1row(lab[k], v) for k, v in R["robust"].items()]
rows.append(p1row("Residual after US projection", R["P3"]["resid_P1"]))
open(T + "tab2_p1.tex", "w").write("\n".join(rows))

# Table 3: structural models
M = E["A2_models"]
order = ["z2 ~ H", "z2 ~ U", "z2 ~ Ufut", "z2 ~ H + U", "z2 ~ H + Ufut"]
def cell(m, v):
    if v not in m["coef"]: return "", ""
    return f"{m['coef'][v]:.3f}{star(m['p'][v])}".replace("^", "$^").replace("}", "}$") if star(m['p'][v]) else f"{m['coef'][v]:.3f}", f"({m['se'][v]:.3f})"
out = []
for v, lab in [("H", "Calendar hours $H$"), ("U", "NYSE sessions $U$"), ("Ufut", "Futures hours $U^{fut}$"), ("Intercept", "Constant")]:
    c = [cell(M[o], v) for o in order]
    out.append(lab + " & " + " & ".join(x[0] for x in c) + " \\\\")
    out.append(" & " + " & ".join(x[1] for x in c) + " \\\\")
out.append("\\midrule AIC & " + " & ".join(f"{M[o]['aic']:.1f}" for o in order) + " \\\\")
out.append("$R^2$ & " + " & ".join(f"{M[o]['r2']:.3f}" for o in order) + " \\\\")
out.append("$N$ & " + " & ".join(str(M[o]['n']) for o in order) + " \\\\")
open(T + "tab3_struct.tex", "w").write("\n".join(out))

# Table 4: US loadings
L = R["P3"]["loadings"]
labs = [("N_0", "Weeknight, before"), ("W_0", "Weekend, before"), ("N_1", "Weeknight, after"), ("W_1", "Weekend, after")]
out = []
for k, lab in labs:
    c = L[k]
    def g(v): return (f"{c['coef'][v]:.3f}", f"({c['se'][v]:.3f})") if v in c["coef"] else ("", "")
    th, fr, ot = g("us_thu"), g("us_fri"), g("us_other")
    out.append(f"{lab} & {th[0]} & {fr[0]} & {ot[0]} & {c['r2']:.2f} & {c['n']} \\\\")
    out.append(f" & {th[1]} & {fr[1]} & {ot[1]} & & \\\\")
open(T + "tab4_loadings.tex", "w").write("\n".join(out))

# Table 5: stock level
C = pd.DataFrame(D["cells"])
out = []
for dl, lab in [(1, "Dual-listed (16 firms)"), (0, "Domestic (110 firms)")]:
    x = C[C.DL == dl].set_index(["post", "typ"])
    out.append(f"{lab} & {x.loc[(0,'N'),'mean_z2']:.3f} & {x.loc[(0,'S'),'R_mean']:.2f} & {x.loc[(0,'W'),'R_mean']:.2f} & {x.loc[(1,'N'),'mean_z2']:.3f} & {x.loc[(1,'W'),'R_mean']:.2f} & {int(x['n'].sum())} \\\\")
open(T + "tab5_stocks.tex", "w").write("\n".join(out))
sc = D["sunday_contrast_levels"]; sl = D["sunday_contrast_logabs"]; rg = S["reg"]; rl = S["reg_logabs"]
tex = []
for k, lab in [("S:DL", "Sunday night $\\times$ DL (pre)")]:
    tex.append(f"{lab} & {sc[k]['coef']:.3f} & ({sc[k]['se']:.3f}) & {sl[k]['coef']:.3f} & ({sl[k]['se']:.3f}) \\\\")
for k, lab in [("W:Post:DL", "Weekend $\\times$ Post $\\times$ DL"), ("W:DL", "Weekend $\\times$ DL"), ("W:Post", "Weekend $\\times$ Post")]:
    tex.append(f"{lab} & {rg['coef'][k]:.3f} & ({rg['se'][k]:.3f}) & {rl[k]['coef']:.3f} & ({rl[k]['se']:.3f}) \\\\")
open(T + "tab6_stockreg.tex", "w").write("\n".join(tex))
json.dump(dict(Delta=S["Delta"], Delta_se=S["Delta_se"], Delta_ci=S["Delta_ci"], Delta_p=S["Delta_p"], idio=S["idio"], decomp=D["DL_decomposition"]),
          open(T + "stock_summary.json", "w"), indent=1)
# vendor table
V = pd.read_csv(T + "vendor_dow_var.csv", index_col=0)
out = []
for d, r in V.iterrows():
    vy = "---" if pd.isna(r.var_yahoo) else f"{r.var_yahoo:.2f}"; ny = "0" if pd.isna(r.n_yahoo) else str(int(r.n_yahoo))
    out.append(f"{d} & {int(r.n_official)} & {r.var_official:.2f} & {ny} & {vy} \\\\")
open(T + "tab7_vendor.tex", "w").write("\n".join(out))
print("tables written")

# Table 8 (revision): DL gaps on own foreign-listing return, and zero-open margins
V = json.load(open("output/results_revision.json"))
LO = pd.DataFrame(V["dl_foreign_loadings"])
nm = {(0, "N"): "Weeknight, before", (0, "S"): "Sunday night, before", (0, "W"): "Weekend, before", (1, "N"): "Weeknight, after", (1, "W"): "Weekend, after"}
out = []
for _, r in LO.iterrows():
    b = "---" if pd.isna(r.beta) else f"{r.beta:.3f}"; s = "" if pd.isna(r.se) else f"({r.se:.3f})"
    out.append(f"{nm[(int(r.post), r.typ)]} & {int(r.n)} & {r.mean_nf:.2f} & {b} & {s} & {r.r2:.2f} & {r.R:.2f} & {r.R_resid:.2f} \\\\")
open(T + "tab8_dlforeign.tex", "w").write("\n".join(out))
Z = pd.DataFrame(V["zero_margins"])
out = []
for dl, lab in [(1, "Dual-listed"), (0, "Domestic")]:
    for _, r in Z[Z.DL == dl].iterrows():
        out.append(f"{lab} & {nm[(int(r.post), r.typ)]} & {r.p_nonzero:.2f} & {r.R_uncond:.2f} & {r.R_cond:.2f} \\\\")
open(T + "tab9_zeros.tex", "w").write("\n".join(out))
F = pd.DataFrame(V["dl_firms"]).dropna(subset=["R_S"])
first = V["active_DL"]["first_foreign"]
out = [f"{r.symbol} & {r.R_S:.2f} & {r.R_W_pre:.2f} & {r.R_W_post:.2f} & {r.p_nonzero:.2f} & {('before 2020' if first.get(r.sec, '') <= '2019-12-02' else first.get(r.sec, ''))} \\\\" for _, r in F.iterrows()]
open(T + "tab10_dlfirms.tex", "w").write("\n".join(out))
print("revision tables written")
