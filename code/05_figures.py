"""Figures for the paper. Palette validated with the dataviz validator (blue/orange/aqua, light mode)."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

# ---------- Figure 1: timeline of one week in Israel time ----------
def fig_timeline():
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    # hours from Thursday 00:00 Israel time, 4.5 days
    def span(day, h0, h1): return (day * 24 + h0, h1 - h0)
    rows = {"TASE, before (Sun–Thu)": 3.2, "TASE, after (Mon–Fri)": 2.4, "NYSE cash session": 1.4, "CME equity futures": 0.6}
    # TASE pre: Thu 10:00-17:25, Sun 10:00-15:50, Mon 10:00-17:25
    pre = [span(0, 10, 17 + 25 / 60), span(3, 10, 15 + 50 / 60), span(4, 10, 17 + 25 / 60)]
    post = [span(0, 10, 17 + 25 / 60), span(1, 10, 13 + 45 / 60), span(4, 10, 17 + 25 / 60)]
    nyse = [span(0, 16.5, 23), span(1, 16.5, 23), span(4, 16.5, 23)]
    # futures in Israel time (NY+7): trade except 00:00-01:00 IL daily; closed Sat 00:00 IL to Mon 01:00 IL
    fut = [span(0, 0, 24 - 0.001 - 23 + 23)]  # placeholder replaced below
    fut = [(0 + 1, 23), (24 + 1, 23), (4 * 24 + 1, 23)]  # Thu 01-24, Fri 01-24(Sat 00:00), Mon 01-24
    for y, bars, c in [(3.2, pre, BLUE), (2.4, post, ORANGE), (1.4, nyse, MUTED), (0.6, fut, "#c9c8c0")]:
        ax.broken_barh(bars, (y - 0.25, 0.5), facecolors=c, edgecolor="white", linewidth=1)
    # gaps
    ax.annotate("", xy=(3 * 24 + 10, 3.62), xytext=(17 + 25 / 60, 3.62), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9))
    ax.text((17.4 + 82) / 2, 3.72, "weekend gap before: 64.6 h, ~1.8 NYSE sessions", color=INK, ha="center", fontsize=7.5)
    ax.annotate("", xy=(4 * 24 + 10, 1.98), xytext=(24 + 13.75, 1.98), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=0.9))
    ax.text((37.75 + 106) / 2, 1.78, "weekend gap after: 68.2 h, ~1 NYSE session", color=INK, ha="center", fontsize=7.5)
    ax.set_yticks(list(rows.values())); ax.set_yticklabels(list(rows.keys()))
    ax.set_xticks([d * 24 + 12 for d in range(5)]); ax.set_xticklabels(["Thu", "Fri", "Sat", "Sun", "Mon"])
    for d in range(1, 5): ax.axvline(d * 24, color="#e4e3dd", lw=0.6, zorder=0)
    ax.set_xlim(0, 5 * 24); ax.set_ylim(0.2, 4.0); ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Israel time (one week around the weekend)", color=INK)
    fig.tight_layout(); fig.savefig("output/figures/fig1_timeline.pdf"); plt.close(fig)

# ---------- Figure 2: relative gap variance by type ----------
def fig_ratios():
    g = pd.read_csv("data/clean/gaps_TA125.csv", parse_dates=["date"])
    g = g[(g.date >= "2020-03-08") & g.typ.isin(["N", "W", "S"])].dropna(subset=["z2"])
    g["week"] = (g.date + pd.Timedelta(days=1)).dt.to_period("W").astype(str)
    rng = np.random.default_rng(1)
    def stat(df, kind):
        n = df[df.typ == "N"].z2
        out = {}
        for t in ("S", "W"):
            x = df[df.typ == t].z2
            if len(x) == 0: continue
            out[t] = x.mean() / n.mean() if kind == "mean" else (np.median(np.sqrt(x)) / np.median(np.sqrt(n))) ** 2
        return out
    res = {}
    for p in (0, 1):
        df = g[g.post == p]; weeks = df.week.unique(); grp = {w: x for w, x in df.groupby("week")}
        for kind in ("mean", "median"):
            est = stat(df, kind); bs = {k: [] for k in est}
            for b in range(999):
                ws = rng.choice(weeks, len(weeks)); d = pd.concat([grp[w] for w in ws]); s = stat(d, kind)
                for k in est: bs[k].append(s[k])
            for k in est: res[(p, kind, k)] = (est[k], *np.percentile(bs[k], [2.5, 97.5]))
    json.dump({str(k): v for k, v in res.items()}, open("output/fig2_values.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    labels = [("S", 0, "Sunday night\n(before)"), ("W", 0, "Weekend\n(before)"), ("W", 1, "Weekend\n(after)")]
    pred = {"cal": {("S", 0): 18.17 / 16.58, ("W", 0): 64.58 / 16.58, ("W", 1): 68.25 / 16.58},
            "fut": {("S", 0): 9.07 / 15.75, ("W", 0): 29.68 / 15.75, ("W", 1): 19.28 / 15.74}}
    for ax, kind, title in [(axes[0], "mean", "Mean-based ratio (pre-specified)"), (axes[1], "median", "Median-based ratio (exploratory)")]:
        for i, (t, p, lab) in enumerate(labels):
            e, lo, hi = res[(p, kind, t)]
            c = BLUE if p == 0 else ORANGE
            ax.bar(i, e, width=0.55, color=c, edgecolor="white")
            ax.errorbar(i, e, yerr=[[e - lo], [hi - e]], color=INK, lw=0.8, capsize=2.5)
            ax.plot([i - 0.33, i + 0.33], [pred["cal"][(t, p)]] * 2, color=INK, lw=1.0, ls="--")
            ax.plot([i - 0.33, i + 0.33], [pred["fut"][(t, p)]] * 2, color=AQUA, lw=1.4, ls=":")
            ax.text(i + 0.30, e + 0.05, f"{e:.2f}", ha="left", va="bottom", fontsize=7.5, color=INK)
        ax.axhline(1, color=MUTED, lw=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels([l[2] for l in labels], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK)
    axes[0].set_ylabel("variance relative to ordinary weeknight")
    axes[1].plot([], [], color=INK, ls="--", label="calendar-hours prediction")
    axes[1].plot([], [], color=AQUA, ls=":", lw=1.4, label="futures-hours prediction")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("output/figures/fig2_ratios.pdf"); plt.close(fig)

# ---------- Figure 3: vendor exhibit ----------
def fig_vendor():
    off = pd.read_csv("data/raw/tase/tase_TA125_official.csv", parse_dates=["date"]).set_index("date")["close"]
    yh = pd.read_csv("data/raw/tase/TA125.TA.csv", index_col=0, parse_dates=True)["Close"]
    a, b = "2019-01-01", "2025-12-31"
    order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    def v(s):
        s = s[(s.index >= a) & (s.index <= b)]; r = np.log(s).diff().dropna() * 100
        return r.groupby(r.index.day_name()).var().reindex(order), r.groupby(r.index.day_name()).size().reindex(order)
    vo, no = v(off); vy, ny = v(yh)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    x = np.arange(5)
    ax.bar(x - 0.18, vo.values, 0.34, color=BLUE, edgecolor="white", label="official TASE record")
    ax.bar(x + 0.18, np.nan_to_num(vy.values), 0.34, color=ORANGE, edgecolor="white", label="Yahoo Finance ^TA125.TA")
    ax.text(0 + 0.18, 0.05, "no\nrows", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([o[:3] for o in order])
    ax.set_ylabel("variance of daily return (%²)"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig("output/figures/fig3_vendor.pdf"); plt.close(fig)
    pd.DataFrame({"var_official": vo, "n_official": no, "var_yahoo": vy, "n_yahoo": ny}).to_csv("output/tables/vendor_dow_var.csv")

def fig_stocks():
    C = pd.read_csv("output/tables/stock_cells.csv")
    D = json.load(open("output/results_stock_decomposition.json"))["DL_decomposition"]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    cats = [(0, "S", "Sunday night\n(before)"), (0, "W", "Weekend\n(before)"), (1, "W", "Weekend\n(after)")]
    for j, (dl, c, lab) in enumerate([(1, BLUE, "dual-listed"), (0, ORANGE, "domestic")]):
        for i, (p, t, _) in enumerate(cats):
            v = C[(C.DL == dl) & (C.post == p) & (C.typ == t)].R_mean.iloc[0]
            x = i + (j - 0.5) * 0.36
            ax.bar(x, v, 0.34, color=c, edgecolor="white", label=lab if i == 0 else None)
            ax.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    x = 2 - 0.18
    ax.plot([x - 0.17, x + 0.17], [D["pred_post"]] * 2, color=INK, ls="--", lw=1)
    ax.plot([x - 0.17, x + 0.17], [D["pred_pure_info"]] * 2, color=AQUA, ls=":", lw=1.6)
    ax.plot([], [], color=INK, ls="--", label="prediction: own-trading + weekend terms")
    ax.plot([], [], color=AQUA, ls=":", lw=1.6, label="prediction: own trading time only")
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in cats], fontsize=7.5)
    ax.set_ylabel("relative gap variance"); ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylim(0, 3.2)
    fig.tight_layout(); fig.savefig("output/figures/fig4_stocks.pdf"); plt.close(fig)

if __name__ == "__main__":
    import os; os.makedirs("output/figures", exist_ok=True)
    fig_timeline(); fig_ratios(); fig_vendor(); fig_stocks(); print("ok")
