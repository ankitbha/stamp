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
import json
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


def panel_exp4(ax):  # wind x layout: sigma_J vs coherence, all 18 cells of evaluation/iasa_pol/runs/exp04_seed0
    r = _load("evaluation/iasa_pol/runs/exp04_seed0/result.json")
    r = r.get("result", r)
    short = {"regulatory": "reg", "random": "rand", "downwind": "down"}
    rows = [(row["wind"], short[row["layout"]], row["sigma_J"], row["max_eligible_coherence"],
             row["coefficient_relative_error"]) for row in r["rows"]]
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
    ax.set_title("(f) Exp: lag window")
    axr = ax.twinx()
    l2 = axr.plot(L, kappa, "D--", color="#c0392b", lw=1.0, ms=3, label=r"$\kappa$ (log)")
    axr.set_yscale("log"); axr.set_ylabel(r"$\kappa$ (log)", color="#c0392b")
    axr.tick_params(axis="y", colors="#c0392b"); axr.grid(False)
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
    scen = ["base", "loc.", "scale", "alt.", "swap"]
    sigmaJ = [21.56, 28.85, 47.00, 46.01, 21.56]
    x = np.arange(len(scen))
    ax.bar(x, sigmaJ, 0.6, color="#5d6d7e", edgecolor="k", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(scen, rotation=0)
    ax.set_ylabel(r"$\sigma_J$"); ax.set_title("(h) Exp: inventory robust.")
    ax.text(0.5, 0.93, "coef. err $=0$ (exact)", transform=ax.transAxes,
            fontsize=5, ha="center", va="center", color="k",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))


def _load(rel):
    with open(os.path.join(HERE, "..", "..", rel)) as f:
        return json.load(f)


def panel_baselines(ax):  # Exp 11 on the signal target (evaluation/iasa_pol/runs_signal_target)
    # Largest relative group-signal error per estimator; background stress over three
    # seeds (points), collapse only in the seed where the plumes reach the sensors.
    methods = [("IASA", "IASA"), ("plain_nnls_B1", "NNLS"), ("cmb_B3", "CMB")]
    stress, collapse = {m: [] for m, _ in methods}, {m: None for m, _ in methods}
    for seed in (0, 1, 2):
        r = _load(f"evaluation/iasa_pol/runs_signal_target/exp11_seed{seed}/result.json")
        r = r.get("result", r)
        for sc in r["scenarios"]:
            st = sc["signal_target"]
            for m, _ in methods:
                err = max(e for e in st["estimators"][m]["reported_group_relative_signal_error"] if e is not None)
                if sc["scenario"] == "background_stress":
                    stress[m].append(err)
                elif max(st["H_tilde_column_norms"]) > 1e-6:
                    collapse[m] = err
    # Plain NNLS reports the two steady-wind sources separately, so it is scored per
    # source (largest of the two) rather than on IASA's merged block.  The saved seed-2
    # arrays hold the projected NNLS solution; c_true = (1.0, 0.7) as in the experiment.
    a = np.load(os.path.join(HERE, "..", "..", "evaluation/iasa_pol/runs_signal_target/exp11_seed2/arrays.npz"))
    H, c_hat, c_true = a["H_tilde"].astype(float), a["c_hat"].astype(float), np.array([1.0, 0.7])
    collapse["plain_nnls_B1"] = max(np.linalg.norm(H[:, k] * (c_hat[k] - c_true[k])) / np.linalg.norm(H[:, k] * c_true[k])
                                    for k in range(2))
    x = np.arange(len(methods)); w = 0.36
    for i, (m, _) in enumerate(methods):
        ax.scatter(np.full(3, x[i] + w / 2), stress[m], s=7, color="#e67e22", zorder=3,
                   label="bg. stress (3 seeds)" if i == 0 else None)
        ax.bar(x[i] - w / 2, collapse[m], w, color="#8e44ad", label="steady wind (seed 2)" if i == 0 else None)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 30)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nflag " + (r"$\checkmark$" if lab == "IASA" else r"$\times$") for _, lab in methods])
    ax.set_ylabel("group-signal rel. err (log)")
    ax.set_title("(a) Exp: baselines")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1, frameon=False,
              fontsize=5.2, handletextpad=0.3, labelspacing=0.15)


def panel_obs_geometry(ax):  # separation of each source group per week (evaluation/iasa_pol/signal_target/weeks.json)
    r = _load("evaluation/iasa_pol/signal_target/weeks.json")
    names = [("brick_kilns", "brick", "#b7472a"), ("industries", "industry", "#7f8c8d"),
             ("population_density", "popul.", "#2e86c1"), ("traffic", "traffic", "#f1c40f")]
    x = np.arange(4); w = 0.2
    for i, (key, lab, col) in enumerate(names):
        vals = [wk["groups"][key]["separation"] for wk in r["weeks"]]
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=lab)
    ax.axhline(r["tau_theta"], color="k", lw=0.6, ls="--")
    ax.set_ylim(0, 1.0); ax.set_ylabel(r"separation $s_g$")
    ax.set_xticks(x); ax.set_xticklabels(["1", "2", "3", "4"]); ax.set_xlabel("week")
    ax.set_title("(a) Obs. separation")


def panel_obs_signal(ax):  # fitted group-signal norm (bars) and its error bound b_g at sigma_e, delta (ticks)
    r = _load("evaluation/iasa_pol/signal_target/weeks.json")
    names = [("brick_kilns", "brick", "#b7472a"), ("industries", "industry", "#7f8c8d"),
             ("population_density", "popul.", "#2e86c1"), ("traffic", "traffic", "#f1c40f")]
    x = np.arange(4); w = 0.2
    for i, (key, lab, col) in enumerate(names):
        xs = x + (i - 1.5) * w
        vals = [wk["groups"][key]["signal_norm"] for wk in r["weeks"]]
        bnds = [wk["groups"][key]["bounds"]["iid_nnls"]["error_norm_bound"] for wk in r["weeks"]]
        ax.bar(xs, vals, w, color=col, label=lab)
        ax.scatter(xs, bnds, marker="_", s=30, color="k", linewidths=0.9, zorder=3,
                   label=r"bound $b_g$" if i == 0 else None)
    ax.set_ylabel(r"norm ($\mu$g/m$^3$)")
    ax.set_xticks(x); ax.set_xticklabels(["1", "2", "3", "4"]); ax.set_xlabel("week")
    ax.set_title("(b) Obs. signal vs. bound")
    ax.legend(loc="upper center", bbox_to_anchor=(-0.45, -0.40), ncol=5, frameon=False,
              columnspacing=0.6, handletextpad=0.3, labelspacing=0.2)


# --------------------------------------------------------------------------- #
# Figure assembly                                                             #
# --------------------------------------------------------------------------- #
def _relabel(ax, letter):
    t = ax.get_title()
    ax.set_title(f"({letter})" + t[t.index(")") + 1:])


def fig_controlled():  # noqa: D401
    # Double-column, 2x4: (a) baselines, (b) conditioning, (c) wind x layout, (d) background
    # stress; (e) transport error, (f) lag window, (g) temporal basis, (h) inventory version.
    fig, axes = plt.subplots(
        2, 4, figsize=(7.15, 3.55),
        gridspec_kw=dict(wspace=0.74, hspace=0.95, left=0.055, right=0.96, top=0.93, bottom=0.10))
    panel_baselines(axes[0, 0]); panel_exp1(axes[0, 1]); panel_exp4(axes[0, 2]); panel_exp3(axes[0, 3])
    panel_exp5(axes[1, 0]); panel_exp7(axes[1, 1]); panel_exp9(axes[1, 2]); panel_exp6(axes[1, 3])
    out = os.path.join(HERE, "fig_controlled.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_observed():
    # Single-column, two observed New Delhi panels side by side.
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(3.35, 1.55),
        gridspec_kw=dict(wspace=1.25, left=0.11, right=0.97, top=0.89, bottom=0.40))
    panel_obs_geometry(axA); panel_obs_signal(axB)
    out = os.path.join(HERE, "fig_observed.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    fig_controlled()
    fig_observed()
