#!/usr/bin/env python3
"""Generate the two main-text results figures for the AAAI 2027 paper.

  fig_controlled.pdf          -- seven controlled-experiment panels (Exp 1,4,3,5,7,9,6)
  fig_baselines_observed.pdf  -- baseline comparison (Exp 11) + observed New Delhi

Every value is transcribed verbatim from the evaluation result tables in the
appendix (tab:results_*) and the committed exp11 run, so the figures are a faithful
re-encoding of those tables. Every legend is placed outside the data area. Run
in-container:

  singularity exec --overlay overlay-25GB-500K.ext3:ro <sif> \
    /bin/bash -lc "source /ext3/env.sh && cd paper/figures && python3 make_figures.py"
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


# --------------------------------------------------------------------------- #
# Individual panel painters (each takes an Axes)                              #
# --------------------------------------------------------------------------- #
def panel_exp1(ax):  # conditioning sets the recovery ceiling (tab:results_h1)
    noise = [0.00, 0.05, 0.10, 0.20]
    sep = [0.000, 0.066, 0.132, 0.264]   # separated: sigma_J=3.27, kappa=8.89
    clo = [0.000, 0.028, 0.057, 0.113]   # close:     sigma_J=9.60, kappa=2.26
    ax.plot(noise, sep, "o-", color="#c0392b", lw=1.2, ms=3, label="separated")
    ax.plot(noise, clo, "s-", color="#2471a3", lw=1.2, ms=3, label="close")
    ax.legend(loc="upper left", frameon=False, fontsize=5, handletextpad=0.3,
              labelspacing=0.2, borderpad=0.1, handlelength=1.4)
    ax.set_xlabel("obs. noise"); ax.set_ylabel("coef. rel. error")
    ax.set_title("(b) Exp: conditioning")
    ax.set_xlim(-0.01, 0.235); ax.set_xticks([0.0, 0.1, 0.2])


def panel_exp4(ax):  # wind x layout: sigma_J vs coherence (tab:results_h4)
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
        ax.scatter(coh, max(sj, 1e-3), s=12 + 340 * err, color=cmap[wind],
                   marker=markers[lay], edgecolor="k", linewidth=0.3, alpha=0.85, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("max coherence"); ax.set_ylabel(r"$\sigma_J$ (log)")
    ax.set_title(r"(c) Exp: wind$\times$layout"); ax.set_xticks([0.0, 0.5, 1.0])
    ax.annotate("single/\nrandom", xy=(1.0, 1e-3), xytext=(0.34, 0.02),
                fontsize=4.6, arrowprops=dict(arrowstyle="->", lw=0.4))
    # (c) has no in-panel legend room; place one combined key in the strip below it.
    # Column 1 = layout (marker), columns 2-3 = wind provider (colour), column-major.
    layout_handles = [Line2D([], [], marker=m, color="0.35", ls="none", ms=4,
                             markeredgecolor="k", markeredgewidth=0.3, label=l)
                      for l, m in [("regulatory", "o"), ("random", "s"), ("downwind", "^")]]
    wind_handles = [Line2D([], [], marker="s", color=cmap[w], ls="none", ms=4,
                           markeredgecolor="k", markeredgewidth=0.3, label=w)
                    for w in winds]
    ax.legend(handles=layout_handles + wind_handles, loc="upper center",
              bbox_to_anchor=(0.42, -0.28), ncol=3, frameon=False, fontsize=4.4,
              columnspacing=0.8, handletextpad=0.15, labelspacing=0.3,
              title="marker: layout   colour: wind", title_fontsize=4.4)


def panel_exp3(ax):  # background stress (tab:results_h3)
    bgs = ["none", "prim.", "redun.", "stress"]
    minvis = [1.000, 0.972, 0.972, 0.000]
    absorp = [0.000, 0.235, 0.235, 1.000]
    sigmaJ = [0.080, 0.080, 0.080, 0.000]
    x = np.arange(len(bgs)); w = 0.26
    ax.bar(x - w, minvis, w, label="vis.", color="#27ae60")
    ax.bar(x, absorp, w, label="absorp.", color="#e67e22")
    ax.bar(x + w, sigmaJ, w, label=r"$\sigma_J$", color="#8e44ad")
    ax.set_xticks(x); ax.set_xticklabels(bgs, rotation=25, ha="right")
    ax.set_title("(d) Exp: bg. stress"); ax.set_ylabel("value")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
              frameon=False, fontsize=4.8, columnspacing=0.6, handletextpad=0.2, handlelength=1.0)


def panel_exp5(ax):  # transport error: coef err vs operator-error norm (tab:results_h5a)
    direction = ([0.00, 0.29, 0.60, 0.87], [0.00, 0.62, 0.80, 0.86])
    speed = ([0.29, 0.53], [0.45, 0.27])
    dispersion = ([0.19, 0.35], [0.50, 0.89])
    ax.plot(*direction, "o-", color="#c0392b", lw=1.1, ms=3, label="direction")
    ax.plot(*speed, "s", color="#2471a3", ms=3.5, label="speed")
    ax.plot(*dispersion, "^", color="#16a085", ms=3.5, label="dispersion")
    ax.set_xlabel("operator err. norm"); ax.set_ylabel("coef. rel. error")
    ax.set_title("(e) Exp: transport err.")
    ax.legend(loc="lower right", frameon=False, fontsize=4.6, handletextpad=0.2,
              borderpad=0.1, labelspacing=0.2)


def panel_exp7(ax):  # lag-window selection (tab:results_lag)
    L = [4, 6, 8, 10, 12, 16]
    sigmaJ = [0.08, 0.71, 2.46, 4.57, 6.16, 9.07]
    kappa = [569.2, 63.8, 18.6, 10.1, 7.5, 5.1]
    l1 = ax.plot(L, sigmaJ, "o-", color="#2c3e50", lw=1.2, ms=3, label=r"$\sigma_J$")
    ax.set_xlabel("lag window $L$"); ax.set_ylabel(r"$\sigma_J$")
    ax.set_title("(f) Exp: lag selection")
    axr = ax.twinx()
    l2 = axr.plot(L, kappa, "D--", color="#c0392b", lw=1.0, ms=3, label=r"$\kappa$ (log)")
    axr.set_yscale("log"); axr.set_ylabel(r"$\kappa$ (log)", color="#c0392b")
    axr.tick_params(axis="y", colors="#c0392b"); axr.grid(False)
    ax.axvline(16, color="#7f8c8d", lw=0.7, ls=":")
    ax.text(15.3, 3.0, "sel. $L{=}16$", fontsize=4.8, ha="center", va="center",
            rotation=90, color="k",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.legend(l1 + l2, [h.get_label() for h in l1 + l2], loc="center left",
              bbox_to_anchor=(0.0, 0.63), frameon=False, fontsize=4.6,
              handletextpad=0.3, labelspacing=0.2)


def panel_exp9(ax):  # temporal-basis recovery (tab:results_temporal)
    noise = [0.00, 0.02, 0.05, 0.10, 0.20]
    coef = [0.000, 0.132, 0.329, 0.659, 0.900]
    activity = [0.000, 0.075, 0.188, 0.375, 0.582]
    ax.plot(noise, coef, "o-", color="#c0392b", lw=1.2, ms=3, label="coef.")
    ax.plot(noise, activity, "s-", color="#27ae60", lw=1.2, ms=3, label="activity")
    ax.set_xlabel("obs. noise"); ax.set_ylabel("rel. error")
    ax.set_title("(g) Exp: temporal basis")
    ax.legend(loc="upper left", frameon=False, fontsize=4.8, handletextpad=0.3, labelspacing=0.2)


def panel_exp6(ax):  # inventory robustness: sigma_J tracks version (tab:results_h5b)
    scen = ["base", "loc.", "rescale", "alt.\nmap", "swap"]
    sigmaJ = [21.56, 28.85, 47.00, 46.01, 21.56]
    x = np.arange(len(scen))
    ax.bar(x, sigmaJ, 0.6, color="#5d6d7e", edgecolor="k", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(scen, rotation=0)
    ax.set_ylabel(r"$\sigma_J$"); ax.set_title("(h) Exp: inventory robust.")
    ax.text(0.5, 0.45, "coef. err $=0$ (exact)", transform=ax.transAxes,
            fontsize=5, ha="center", va="center", color="k",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


def panel_baselines(ax):  # Exp 11: IASA vs baselines (evaluation/.../exp11_seed0)
    # share-L2 error per method x scenario (floored for the log axis); flag per method.
    methods = ["IASA", "NNLS", "CMB", "PMF"]           # B1=NNLS, B3=CMB, B2=PMF
    collapse = [5e-9, 5e-9, 1.414, 0.0845]
    stress = [0.00606, 1.231, 0.1885, 1.152]
    flag = [True, False, False, False]
    floor = 3e-3
    c = [max(v, floor) for v in collapse]; s = [max(v, floor) for v in stress]
    x = np.arange(len(methods)); w = 0.38
    ax.bar(x - w / 2, c, w, label="collapse", color="#8e44ad")
    ax.bar(x + w / 2, s, w, label="bg. stress", color="#e67e22")
    ax.set_yscale("log"); ax.set_ylim(floor, 3.0)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("apportion. err (log)")
    ax.set_title("(a) Exp: baselines")
    # identifiability flag row under the method labels
    for xi, fl in zip(x, flag):
        ax.annotate("flag\n" + (r"$\checkmark$" if fl else r"$\times$"),
                    xy=(xi, 0.11), xycoords="data", xytext=(0, -16),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=6, fontweight="bold", color="k",
                    bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.75))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
              fontsize=4.8, columnspacing=0.8, handletextpad=0.3)


def panel_obs_geometry(ax):  # observed weeks 1-4 geometry (tab:results_nd_ident)
    weeks = [1, 2, 3, 4]
    sig1 = [50.1, 60.3, 77.0, 81.2]; sigJ = [3.71, 5.67, 10.15, 7.65]
    coh = [0.316, 0.355, 0.496, 0.475]
    x = np.arange(len(weeks)); w = 0.34
    b1 = ax.bar(x - w / 2, sig1, w, label=r"$\sigma_1$", color="#2c3e50")
    b2 = ax.bar(x + w / 2, sigJ, w, label=r"$\sigma_J$", color="#5dade2")
    ax.set_yscale("log"); ax.set_ylabel("sing. val. (log)")
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in weeks]); ax.set_xlabel("week")
    ax.set_title("(a) Obs. geometry")
    axr = ax.twinx()
    ln = axr.plot(x, coh, "D--", color="#c0392b", ms=3, lw=1.0, label="max coh.")
    axr.set_ylim(0, 1); axr.tick_params(axis="y", colors="#c0392b"); axr.grid(False)
    ax.legend([b1, b2, ln[0]], [r"$\sigma_1$", r"$\sigma_J$", "coh."],
              loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=3,
              frameon=False, columnspacing=0.8, handletextpad=0.3)


def panel_obs_apportion(ax):  # observed apportionment (tab:results_nd_appt)
    weeks = [1, 2, 3, 4]; x = np.arange(len(weeks))
    series = [("brick", [0.00, 0.00, 0.06, 0.93], "#b7472a"),
              ("industry", [0.00, 0.16, 0.00, 0.07], "#7f8c8d"),
              ("popul.", [1.00, 0.81, 0.94, 0.00], "#2e86c1"),
              ("traffic", [0.00, 0.04, 0.00, 0.00], "#f1c40f")]
    bottom = np.zeros(len(weeks))
    for name, vals, col in series:
        ax.bar(x, vals, 0.6, bottom=bottom, label=name, color=col)
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in weeks]); ax.set_xlabel("week")
    ax.set_ylabel("frac. of signal"); ax.set_ylim(0, 1.0)
    ax.set_title("(b) Obs. apportion.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=2, frameon=False,
              columnspacing=0.6, handletextpad=0.3, labelspacing=0.2)


# --------------------------------------------------------------------------- #
# Figure assembly                                                             #
# --------------------------------------------------------------------------- #
def fig_controlled():
    # Double-column, 2x4: baselines top-left (a), then the seven diagnostics (b-h).
    fig, axes = plt.subplots(
        2, 4, figsize=(7.15, 3.75),
        gridspec_kw=dict(wspace=0.74, hspace=0.75, left=0.055, right=0.96,
                         top=0.93, bottom=0.13))
    panel_baselines(axes[0, 0]); panel_exp1(axes[0, 1]); panel_exp4(axes[0, 2]); panel_exp3(axes[0, 3])
    panel_exp5(axes[1, 0]); panel_exp7(axes[1, 1]); panel_exp9(axes[1, 2]); panel_exp6(axes[1, 3])
    out = os.path.join(HERE, "fig_controlled.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_observed():
    # Single-column, two observed New Delhi panels side by side.
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(3.35, 1.58),
        gridspec_kw=dict(wspace=1.05, left=0.13, right=0.89, top=0.88, bottom=0.42))
    panel_obs_geometry(axA); panel_obs_apportion(axB)
    out = os.path.join(HERE, "fig_observed.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    fig_controlled()
    fig_observed()
