"""Figures and LaTeX tables for the three-switch manuscript, from derived results only.
Input  output/results_gcc.json, output/results_gcc_exploratory.json, output/gcc_calendar_summary.csv, output/results*.json (TASE)
Output output/figures/fig_ratios.pdf, fig_switch.pdf; output/tables/t_*.tex
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
r = json.load(open("output/results_gcc.json")); x = json.load(open("output/results_gcc_exploratory.json"))
MK = [("KSA", "Saudi Arabia"), ("UAE", "UAE (DFM + ADX)"), ("TASE_domestic", "Tel Aviv, domestic"), ("TASE_dual", "Tel Aviv, dual-listed")]
REG = {"KSA": ("Sat–Wed", "Sun–Thu"), "UAE": ("Sun–Thu", "Mon–Fri"), "TASE_domestic": ("Sun–Thu", "Mon–Fri"), "TASE_dual": ("Sun–Thu", "Mon–Fri")}

def fig_ratios():
    rows = []
    for k, lab in MK:
        for p in (0, 1):
            rows.append((k, lab, p))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab, p) in zip(y, rows):
        R = r[k]["R"]
        for t, c, off in (("W", BLUE, 0.12), ("Z", ORANGE, -0.12)):
            key = f"{t}_post{p}"
            if key not in R: continue
            v = R[key]; lo, hi = np.exp(v["logR"] - 1.96 * v["se"]), np.exp(v["logR"] + 1.96 * v["se"])
            ax.plot([lo, hi], [yi + off] * 2, color=c, lw=1.2)
            ax.plot(v["R"], yi + off, "o", ms=5, color=c, mec="white", mew=0.8)
            if t == "W":
                ax.plot(v["calendar_R"], yi + off, "|", ms=9, mew=1.4, color=INK)
    ax.axvline(1, color=MUTED, lw=0.6)
    ax.set_xscale("log"); ax.set_xticks([0.4, 0.6, 1, 1.5, 2, 3, 4]); ax.set_xticklabels(["0.4", "0.6", "1", "1.5", "2", "3", "4"])
    ax.set_yticks(y); ax.set_yticklabels([f"{lab}, {REG[k][p]}" for k, lab, p in rows], fontsize=7.8)
    ax.set_xlabel("variance of the closure gap relative to an ordinary weeknight (log scale)")
    ax.plot([], [], "o", color=BLUE, label="weekend closure"); ax.plot([], [], "o", color=ORANGE, label="weeknight without a US session")
    ax.plot([], [], "|", color=INK, ms=9, mew=1.4, label="calendar-time prediction")
    ax.legend(frameon=False, fontsize=7.2, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig("output/figures/fig_ratios.pdf"); plt.close(fig)

def fig_switch():
    b1 = x["B1"]["markets"]
    rows = [("KSA", "Saudi Arabia, 2013"), ("UAE", "UAE, 2022"), ("TASE_domestic", "Tel Aviv domestic, 2026"), ("TASE_dual", "Tel Aviv dual-listed, 2026")]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = np.arange(len(rows) + 1)[::-1]
    for yi, (k, lab) in zip(y[:-1], rows):
        d = r[k]["DeltaW"]
        ax.plot([d["est"] - 1.96 * d["se"], d["est"] + 1.96 * d["se"]], [yi] * 2, color=BLUE, lw=1.2)
        ax.plot(d["est"], yi, "o", color=BLUE, ms=5, mec="white", mew=0.8)
        ax.plot(b1[k]["pred_DeltaW_from_weeknight"], yi, "D", color=AQUA, ms=5, mec="white", mew=0.6)
        ax.plot(d["calendar_pred"], yi, "|", color=INK, ms=9, mew=1.4)
    c = r["DeltaW_combined"]
    ax.plot([c["est"] - 1.96 * c["se"], c["est"] + 1.96 * c["se"]], [y[-1]] * 2, color=INK, lw=1.6)
    ax.plot(c["est"], y[-1], "s", color=INK, ms=5)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in rows] + ["combined (inverse variance)"], fontsize=7.8)
    ax.set_xlabel("change in log weekend variance ratio at the switch")
    ax.plot([], [], "o", color=BLUE, label="observed, 95% interval"); ax.plot([], [], "D", color=AQUA, label="predicted from the weeknight contrast")
    ax.plot([], [], "|", color=INK, ms=9, mew=1.4, label="calendar-time prediction")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig("output/figures/fig_switch.pdf"); plt.close(fig)

def f(v, d=2): return f"{v:.{d}f}"
def tables():
    cal = pd.read_csv("output/gcc_calendar_summary.csv")
    # Table: closure types by market and regime (Gulf from calendar; TASE from cells)
    lines = []
    names = {"TASI": "Saudi Arabia", "DFMGI": "DFM", "FTFADGI": "ADX"}
    for m in ("TASI", "DFMGI", "FTFADGI"):
        for p in (0, 1):
            c = cal[(cal.market == m) & (cal.post == p)].set_index("typ")
            cells = []
            for t in ("N", "Z", "W"):
                cells += [f"{int(c.loc[t,'n'])}", f(c.loc[t, "H"], 1), f(c.loc[t, "U"])] if t in c.index else ["--", "--", "--"]
            lines.append(f"{names[m]} & {['before','after'][p]} & " + " & ".join(cells) + r" \\")
    ta = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date"]); ta = ta[ta.date >= "2020-03-08"]
    ta["typ"] = ta.typ.replace({"S": "Z"})
    for p in (0, 1):
        c = ta[ta.post == p].groupby("typ").agg(n=("H", "size"), H=("H", "mean"), U=("U", "mean"))
        cells = []
        for t in ("N", "Z", "W"):
            cells += [f"{int(c.loc[t,'n'])}", f(c.loc[t, "H"], 1), f(c.loc[t, "U"])] if t in c.index else ["--", "--", "--"]
        lines.append(f"Tel Aviv & {['before','after'][p]} & " + " & ".join(cells) + r" \\")
    open("output/tables/t_calendar.tex", "w").write("\n".join(lines) + "\n")
    # Table: ratios and tests
    L = []
    for k, lab in MK:
        R = r[k]["R"]; D = r[k]["DeltaW"]
        def cell(key):
            if key not in R: return "--", ""
            v = R[key]; return f(v["R"]), f"({f(v['se'])})"
        z0, z0s = cell("Z_post0"); w0, w0s = cell("W_post0"); z1, z1s = cell("Z_post1"); w1, w1s = cell("W_post1")
        L.append(f"{lab} & {z0} & {w0} & {z1} & {w1} & {f(D['est'])} & {f(D['calendar_pred'])} & {r[k]['firms']} \\\\")
        L.append(f" & {z0s} & {w0s} & {z1s} & {w1s} & ({f(D['se'])}) & & \\\\")
    c = r["DeltaW_combined"]
    L.append(r"\midrule" + f"\nCombined & & & & & {f(c['est'])} & & \\\\\n & & & & & ({f(c['se'])}) & & \\\\")
    open("output/tables/t_ratios.tex", "w").write("\n".join(L) + "\n")
    # Table: robustness of the Gulf estimates
    rob = [("Primary", r["UAE"], r["KSA"]),
           ("Unwinsorised", r["UAE_unwinsorised"], r["KSA_unwinsorised"]),
           ("No Covid crash / $\\pm$2 years", r["UAE_drop_covid"], r["KSA_pm2"]),
           ("Symmetric / $\\pm$5 years", r["UAE_symmetric"], r["KSA_pm5"])]
    def g(res, key):
        if key == "DeltaW": return f(res["DeltaW"]["est"]) + f" ({f(res['DeltaW']['se'])})"
        return f(res["R"][key]["R"])
    ex = x
    def gx(d, key):
        if key == "DeltaW": return f(d["DeltaW"]["est"]) + f" ({f(d['DeltaW']['se'])})"
        return f(d[key]["R"])
    L = []
    for lab, u, k in rob:
        L.append(f"{lab} & {g(u,'Z_post0')} & {g(u,'Z_post1')} & {g(u,'DeltaW')} & {g(k,'Z_post0')} & {g(k,'Z_post1')} & {g(k,'DeltaW')} \\\\")
    L.append(r"\midrule")
    exr = [("No 24 Dec--2 Jan$^{\dagger}$", ex["B2_no_holiday_season"]["UAE"], ex["B2_no_holiday_season"]["KSA"]),
           ("66-session scale$^{\dagger}$", ex["B4_s66"]["UAE"], ex["B4_s66"]["KSA"]),
           ("No Ramadan$^{\dagger}$", ex["B6_no_ramadan"]["UAE"], ex["B6_no_ramadan"]["KSA"])]
    for lab, u, k in exr:
        L.append(f"{lab} & {gx(u,'Z_post0')} & {gx(u,'Z_post1')} & {gx(u,'DeltaW')} & {gx(k,'Z_post0')} & {gx(k,'Z_post1')} & {gx(k,'DeltaW')} \\\\")
    k5 = ex["B5_KSA_no_limit_gaps"]; u3 = ex["B3_UAE_denominator_without_Sundays"]
    L.append(f"Scale without Sundays$^{{\\dagger}}$ & {gx(u3,'Z_post0')} & {gx(u3,'Z_post1')} & {gx(u3,'DeltaW')} & & & \\\\")
    L.append(f"No limit gaps$^{{\\dagger}}$ & & & & {gx(k5,'Z_post0')} & {gx(k5,'Z_post1')} & {gx(k5,'DeltaW')} \\\\")
    open("output/tables/t_robust.tex", "w").write("\n".join(L) + "\n")
    # pooled + bound
    b1 = x["B1"]
    def row(lab, P): return f"{lab} & {f(P['b']['est'],3)} & ({f(P['b']['se'],3)}) & {f(P['c']['est'],3)} & ({f(P['c']['se'],3)}) & {P['n']:,} \\\\"
    L = [row("$z^2$ units (pre-specified)", r["pooled_prespecified"]),
         row("Weeknight units", r["pooled"]),
         row("Weeknight units, market $\\times$ calendar", r["pooled_mr"]),
         row("Weeknight units, Gulf only", r["pooled_gulf"])]
    open("output/tables/t_pooled.tex", "w").write("\n".join(L) + "\n")
    L = []
    for k, lab in MK:
        m = b1["markets"][k]
        L.append(f"{lab} & {f(m['delta_weeknight'])} & ({f(m['se'])}) & {f(m['dU'])} & {f(m['delta_weekend'])} & ({f(m['se_w'])}) \\\\")
    L.append(r"\midrule" + f"\nCombined & & & & {f(b1['delta_weekend_combined'])} & ({f(b1['se'])}) \\\\")
    open("output/tables/t_bound.tex", "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    fig_ratios(); fig_switch(); tables(); print("ok")

def table_robust_scale():
    c = json.load(open("output/results_review_checks.json")); L = []
    for k, lab in MK:
        a = c["logabs"][k]; md = c["median_R_nonzero"][k]; tl = c["top1pct_share_of_sum"]
        def g(key): return f(a[key]["R"]) if key in a else "--"
        L.append(f"{lab} & {g('Z_post0')} & {g('W_post0')} & {g('Z_post1')} & {g('W_post1')} & {f(a['DeltaW']['est'])} ({f(a['DeltaW']['se'])}) & {f(a['pred_DeltaW'])} & {f(np.log(md['W_post1']/md['W_post0']))} & {f(tl[k+'|0|W'])} \\\\")
    cb = c["logabs"]["combined"]
    L.append(r"\midrule" + f"\nCombined & & & & & {f(cb['est'])} ({f(cb['se'])}) & & & \\\\")
    open("output/tables/t_robscale.tex", "w").write("\n".join(L) + "\n")

def table_scales():
    a = json.load(open("output/results_abs_scale.json")); r2 = json.load(open("output/results_review_round2.json"))
    ab = r2["scales"]["abs_keep_zeros"]; lg = r2["scales"]["log_nonzero"]; mb = json.load(open("output/results_median_bootstrap.json")); md = mb; tf = r2["tail_frequency"]
    L = []
    SH = {"KSA": "Saudi Arabia", "UAE": "UAE", "TASE_domestic": "TA domestic", "TASE_dual": "TA dual-listed"}
    for k, _ in MK:
        lab = SH[k]; x = a[k]; z = f(x["Z0"]) if "Z0" in x else "--"; z1 = f(x["Z1"]) if "Z1" in x else "--"
        L.append(f"{lab} & {z} & {f(x['W0'])} & {z1} & {f(x['W1'])} & {f(ab['per_market'][k]['est'])} & ({f(ab['per_market'][k]['se'])}) & {f(x['pred'])} & {f(lg['per_market'][k]['est'])} & ({f(lg['per_market'][k]['se'])}) & {f(md[k]['est'])} & ({f(md[k]['se'])}) & {f(tf[k]['DeltaW_log'])} & ({f(tf[k]['se'])}) \\\\")
    ma, ml, mm = ab["meta"], lg["meta"], mb["meta"]
    L.append(r"\midrule")
    L.append(f"Fixed effect & & & & & {f(ma['fixed'])} & ({f(ma['fixed_se'])}) & & {f(ml['fixed'])} & ({f(ml['fixed_se'])}) & {f(mm['fixed'])} & ({f(mm['fixed_se'])}) & & \\\\")
    L.append(f"Random effects & & & & & {f(ma['random'])} & ({f(ma['random_se'])}) & & {f(ml['random'])} & ({f(ml['random_se'])}) & {f(mm['random'])} & ({f(mm['random_se'])}) & & \\\\")
    L.append(f"$I^2$ & & & & & {f(ma['I2'])} & & & {f(ml['I2'])} & & {f(mm['I2'])} & & & \\\\")
    open("output/tables/t_scales.tex", "w").write("\n".join(L) + "\n")
table_scales()

def table_cc():
    c = json.load(open("output/results_close_to_close.json")); a = json.load(open("output/results_abs_scale.json"))
    r2 = json.load(open("output/results_review_round2.json"))["scales"]["abs_keep_zeros"]["per_market"]
    SH = {"KSA": "Saudi Arabia", "UAE": "UAE", "TASE_domestic": "Tel Aviv, domestic", "TASE_dual": "Tel Aviv, dual-listed"}
    L = []
    for k in ("KSA", "UAE", "TASE_domestic", "TASE_dual"):
        gm = r[k]["DeltaW"]; ga = r2[k]; cm = c["mean_scale"][k]["DeltaW"]; ca = c["abs_scale"][k]["DeltaW"]
        L.append(f"{SH[k]} & {f(gm['est'])} & ({f(gm['se'])}) & {f(ga['est'])} & ({f(ga['se'])}) & {f(a[k]['pred'])} & {f(cm['est'])} & ({f(cm['se'])}) & {f(ca['est'])} & ({f(ca['se'])}) \\\\")
    cg = r["DeltaW_combined"]; ab = json.load(open("output/results_review_round2.json"))["scales"]["abs_keep_zeros"]["meta"]
    L.append(r"\midrule")
    L.append(f"Combined & {f(cg['est'])} & ({f(cg['se'])}) & {f(ab['fixed'])} & ({f(ab['fixed_se'])}) & & {f(c['mean_scale']['combined']['est'])} & ({f(c['mean_scale']['combined']['se'])}) & {f(c['abs_scale']['combined']['est'])} & ({f(c['abs_scale']['combined']['se'])}) \\\\")
    open("output/tables/t_cc.tex", "w").write("\n".join(L) + "\n")
table_cc()
