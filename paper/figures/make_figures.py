#!/usr/bin/env python3
"""Generate the single consolidated main-text results figure for the AAAI 2027 paper.

All five panels live in one figure (fig_results.pdf): three controlled panels
(Experiments 1, 4, 3) on the top row and two compact observed New Delhi panels
(weeks 1-4 geometry and apportionment) on the bottom row. Values are transcribed
verbatim from the evaluation-section result tables, so the figure is a faithful
re-encoding of those tables. Every legend is placed outside the data area. Run
in-container:

  singularity exec --overlay overlay-25GB-500K.ext3:ro <sif> \
    /bin/bash -lc "source /ext3/env.sh && cd paper/figures && python3 make_figures.py"
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 6.8,
    "axes.labelsize": 6.5,
    "legend.fontsize": 5,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
HERE = os.path.dirname(os.path.abspath(__file__))


def fig_results():
    fig, (axA, axB, axC, axD, axE) = plt.subplots(
        1, 5, figsize=(7.15, 1.95),
        gridspec_kw=dict(wspace=0.62, left=0.045, right=0.995, top=0.86, bottom=0.34))

    # ---- (a) Exp 1: coefficient error vs noise for two geometries. ----
    noise = [0.00, 0.05, 0.10, 0.20]
    sep = [0.000, 0.066, 0.132, 0.264]   # separated: sigma_J=3.27, kappa=8.89
    clo = [0.000, 0.028, 0.057, 0.113]   # close:     sigma_J=9.60, kappa=2.26
    axA.plot(noise, sep, "o-", color="#c0392b", lw=1.2, ms=3)
    axA.plot(noise, clo, "s-", color="#2471a3", lw=1.2, ms=3)
    axA.text(0.20, 0.268, "separated", color="#c0392b", fontsize=5, ha="right", va="bottom")
    axA.text(0.205, 0.108, "close", color="#2471a3", fontsize=5, ha="right", va="top")
    axA.set_xlabel("obs. noise")
    axA.set_ylabel("coef. rel. error")
    axA.set_title("(a) Exp 1: conditioning")
    axA.set_xlim(-0.01, 0.235)
    axA.set_xticks([0.0, 0.1, 0.2])

    # ---- (b) Exp 4: sigma_J vs max coherence, all 15 wind x layout rows. ----
    # (wind, layout, sigma_J, max_coh, coef_err)
    rows = [
        ("constant", "reg", 1.53, 0.975, 0.000), ("constant", "rand", 0.00, 0.028, 0.000),
        ("constant", "down", 0.13, 0.990, 0.000), ("single", "reg", 6.44, 0.741, 0.000),
        ("single", "rand", 1e-3, 1.000, 0.573),  ("single", "down", 1.36, 0.990, 0.000),
        ("diurnal", "reg", 6.11, 0.722, 0.000),  ("diurnal", "rand", 1e-3, 0.880, 0.001),
        ("ar1", "reg", 7.35, 0.641, 0.000),      ("ar1", "rand", 1e-3, 0.774, 0.115),
        ("multi", "reg", 5.70, 0.129, 0.000),    ("multi", "rand", 2.79, 0.322, 0.000),
        ("real", "reg", 2.85, 0.156, 0.000),     ("real", "rand", 2.10, 0.287, 0.000),
        ("real", "down", 0.38, 0.221, 0.000),
    ]
    winds = ["constant", "single", "diurnal", "ar1", "multi", "real"]
    cmap = dict(zip(winds, plt.cm.viridis(np.linspace(0, 0.9, len(winds)))))
    markers = {"reg": "o", "rand": "s", "down": "^"}
    for wind, lay, sj, coh, err in rows:
        size = 12 + 340 * err     # size encodes coefficient error
        axB.scatter(coh, max(sj, 1e-3), s=size, color=cmap[wind], marker=markers[lay],
                    edgecolor="k", linewidth=0.3, alpha=0.85, zorder=3)
    axB.set_yscale("log")
    axB.set_xlabel("max coherence")
    axB.set_ylabel(r"$\sigma_J$ (log)")
    axB.set_title(r"(b) Exp 4: wind$\times$layout")
    axB.set_xticks([0.0, 0.5, 1.0])
    axB.annotate("single/\nrandom", xy=(1.0, 1e-3), xytext=(0.34, 0.02),
                 fontsize=4.6, arrowprops=dict(arrowstyle="->", lw=0.4))

    # ---- (c) Exp 3: background stress. ----
    bgs = ["none", "prim.", "redun.", "stress"]
    minvis = [1.000, 0.972, 0.972, 0.000]
    absorp = [0.000, 0.235, 0.235, 1.000]
    sigmaJ = [0.080, 0.080, 0.080, 0.000]
    x = np.arange(len(bgs)); w = 0.26
    axC.bar(x - w, minvis, w, label="vis.", color="#27ae60")
    axC.bar(x, absorp, w, label="absorp.", color="#e67e22")
    axC.bar(x + w, sigmaJ, w, label=r"$\sigma_J$", color="#8e44ad")
    axC.set_xticks(x); axC.set_xticklabels(bgs, rotation=25, ha="right")
    axC.set_title("(c) Exp 3: bg. stress")
    axC.set_ylabel("value")
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
               frameon=False, fontsize=4.8, columnspacing=0.6, handletextpad=0.2,
               handlelength=1.0)

    # ---- (d) Observed identifiability geometry (weeks 1-4). ----
    weeks = [1, 2, 3, 4]
    sig1 = [50.1, 60.3, 77.0, 81.2]
    sigJ = [3.71, 5.67, 10.15, 7.65]
    coh = [0.316, 0.355, 0.496, 0.475]
    x = np.arange(len(weeks)); w = 0.34
    b1 = axD.bar(x - w / 2, sig1, w, label=r"$\sigma_1$", color="#2c3e50")
    b2 = axD.bar(x + w / 2, sigJ, w, label=r"$\sigma_J$", color="#5dade2")
    axD.set_yscale("log")
    axD.set_ylabel("sing. val. (log)")
    axD.set_xticks(x); axD.set_xticklabels([str(k) for k in weeks])
    axD.set_xlabel("week")
    axD.set_title("(d) Obs. geometry")
    axr = axD.twinx()
    ln = axr.plot(x, coh, "D--", color="#c0392b", ms=3, lw=1.0, label="max coh.")
    axr.set_ylim(0, 1); axr.tick_params(axis="y", colors="#c0392b")
    axr.grid(False)
    axD.legend([b1, b2, ln[0]], [r"$\sigma_1$", r"$\sigma_J$", "coh."],
               loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
               frameon=False, columnspacing=0.8, handletextpad=0.3)

    # ---- (e) Observed proxy apportionment shares (stacked). ----
    brick = [0.00, 0.00, 0.06, 0.93]
    ind = [0.00, 0.16, 0.00, 0.07]
    pop = [1.00, 0.81, 0.94, 0.00]
    traf = [0.00, 0.04, 0.00, 0.00]
    bottom = np.zeros(len(weeks))
    series = [("brick", brick, "#b7472a"), ("industry", ind, "#7f8c8d"),
              ("popul.", pop, "#2e86c1"), ("traffic", traf, "#f1c40f")]
    for name, vals, col in series:
        axE.bar(x, vals, 0.6, bottom=bottom, label=name, color=col)
        bottom += np.array(vals)
    axE.set_xticks(x); axE.set_xticklabels([str(k) for k in weeks])
    axE.set_xlabel("week")
    axE.set_ylabel("frac. of signal")
    axE.set_ylim(0, 1.0)
    axE.set_title("(e) Obs. apportion.")
    axE.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
               frameon=False, columnspacing=0.6, handletextpad=0.3, labelspacing=0.2)

    out = os.path.join(HERE, "fig_results.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_results()
