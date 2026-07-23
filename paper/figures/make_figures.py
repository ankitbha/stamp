#!/usr/bin/env python3
"""Generate the two consolidated main-text result figures for the AAAI 2027 paper.

Values are transcribed verbatim from the result tables in the evaluation section
(Experiments 1, 3, 4 for Figure 1; the observed New Delhi weeks 1-4 for Figure 2),
so the figures are faithful re-encodings of those tables. Run in-container:

  singularity exec --overlay overlay-25GB-500K.ext3:ro <sif> \
    /bin/bash -lc "source /ext3/env.sh && cd paper/figures && python3 make_figures.py"
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
HERE = os.path.dirname(os.path.abspath(__file__))


def fig1_identifiability():
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.15))

    # -- Panel A: Exp 1 -- coefficient error vs noise for two geometries.
    noise = [0.00, 0.05, 0.10, 0.20]
    sep = [0.000, 0.066, 0.132, 0.264]   # separated: sigma_J=3.27, kappa=8.89
    clo = [0.000, 0.028, 0.057, 0.113]   # close:     sigma_J=9.60, kappa=2.26
    axA.plot(noise, sep, "o-", color="#c0392b", lw=1.4, ms=4,
             label=r"separated ($\sigma_J$=3.27, $\kappa$=8.9)")
    axA.plot(noise, clo, "s-", color="#2471a3", lw=1.4, ms=4,
             label=r"close ($\sigma_J$=9.60, $\kappa$=2.3)")
    axA.set_xlabel("observation noise")
    axA.set_ylabel("coefficient rel. error")
    axA.set_title("(a) Exp 1: conditioning\nsets the recovery ceiling")
    axA.legend(loc="upper left", frameon=False)

    # -- Panel B: Exp 4 -- sigma_J vs max coherence, all 15 rows.
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
        # size encodes coefficient error so the two nonzero-error cases stand out.
        size = 22 + 600 * err
        axB.scatter(coh, max(sj, 1e-3), s=size, color=cmap[wind], marker=markers[lay],
                    edgecolor="k", linewidth=0.3, alpha=0.85, zorder=3)
    axB.set_yscale("log")
    axB.set_xlabel("max eligible coherence")
    axB.set_ylabel(r"$\sigma_J$ (log)")
    axB.set_title("(b) Exp 4: wind $\\times$ layout\n(size $\\propto$ coef. error)")
    axB.annotate("single/random\ncoh 1.0, err 0.57", xy=(1.0, 1e-3), xytext=(0.45, 0.02),
                 fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.5))
    wind_handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[w], mec="k",
                    mew=0.3, ms=5, label=w) for w in winds]
    lay_handles = [plt.Line2D([], [], marker=markers[l], ls="", color="0.6", mec="k",
                   mew=0.3, ms=5, label={"reg": "regulatory", "rand": "random",
                   "down": "downwind"}[l]) for l in markers]
    leg1 = axB.legend(handles=wind_handles, loc="lower left", frameon=False, ncol=2,
                      handletextpad=0.1, columnspacing=0.6)
    axB.add_artist(leg1)
    axB.legend(handles=lay_handles, loc="upper right", frameon=False,
               handletextpad=0.1)

    # -- Panel C: Exp 3 -- background stress.
    bgs = ["none", "primary", "redundant", "stress"]
    minvis = [1.000, 0.972, 0.972, 0.000]
    absorp = [0.000, 0.235, 0.235, 1.000]
    sigmaJ = [0.080, 0.080, 0.080, 0.000]
    x = np.arange(len(bgs)); w = 0.26
    axC.bar(x - w, minvis, w, label="min. vis.", color="#27ae60")
    axC.bar(x, absorp, w, label="absorption", color="#e67e22")
    axC.bar(x + w, sigmaJ, w, label=r"$\sigma_J$", color="#8e44ad")
    axC.set_xticks(x); axC.set_xticklabels(bgs, rotation=20, ha="right")
    axC.set_title("(c) Exp 3: source-like\nbackground erases signal")
    axC.set_ylabel("value")
    axC.legend(loc="center right", frameon=False)

    fig.tight_layout(w_pad=1.0)
    out = os.path.join(HERE, "fig1_identifiability.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def fig2_observed():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    weeks = [1, 2, 3, 4]

    # -- Panel A: observed identifiability geometry (weeks 1-4).
    sig1 = [50.1, 60.3, 77.0, 81.2]
    sigJ = [3.71, 5.67, 10.15, 7.65]
    coh = [0.316, 0.355, 0.496, 0.475]
    x = np.arange(len(weeks)); w = 0.34
    axA.bar(x - w / 2, sig1, w, label=r"$\sigma_1$", color="#2c3e50")
    axA.bar(x + w / 2, sigJ, w, label=r"$\sigma_J$", color="#5dade2")
    axA.set_yscale("log")
    axA.set_ylabel(r"singular value (log)")
    axA.set_xticks(x); axA.set_xticklabels([f"wk{k}" for k in weeks])
    axA.set_title("(a) Observed identifiability\n(all full rank 7, 4 singletons)")
    axA.legend(loc="upper left", frameon=False)
    axr = axA.twinx()
    axr.plot(x, coh, "D--", color="#c0392b", ms=4, lw=1.2, label="max coh.")
    axr.set_ylabel("max eligible coherence", color="#c0392b")
    axr.set_ylim(0, 1); axr.tick_params(axis="y", colors="#c0392b")
    axr.grid(False)
    axr.legend(loc="upper right", frameon=False)

    # -- Panel B: proxy apportionment shares (stacked).
    brick = [0.00, 0.00, 0.06, 0.93]
    ind = [0.00, 0.16, 0.00, 0.07]
    pop = [1.00, 0.81, 0.94, 0.00]
    traf = [0.00, 0.04, 0.00, 0.00]
    bottom = np.zeros(len(weeks))
    series = [("brick kilns", brick, "#b7472a"), ("industries", ind, "#7f8c8d"),
              ("population", pop, "#2e86c1"), ("traffic", traf, "#f1c40f")]
    for name, vals, col in series:
        axB.bar(x, vals, 0.6, bottom=bottom, label=name, color=col)
        bottom += np.array(vals)
    axB.set_xticks(x); axB.set_xticklabels([f"wk{k}" for k in weeks])
    axB.set_ylabel("fraction of fitted sensor signal")
    axB.set_ylim(0, 1.0)
    axB.set_title("(b) Proxy apportionment\n(not physical emissions)")
    axB.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout(w_pad=1.5)
    out = os.path.join(HERE, "fig2_observed.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig1_identifiability()
    fig2_observed()
