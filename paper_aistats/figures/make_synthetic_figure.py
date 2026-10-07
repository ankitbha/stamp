"""Figure for the synthetic study (paper Section 5.1-5.2).

Reads evaluation/iasa_synthetic/results.json and writes fig_synthetic.pdf.
"""

import json
import math
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
RESULTS = os.path.join(HERE, "..", "..", "evaluation", "iasa_synthetic", "results.json")


def panel_conditioning(ax, e1b):
    rows = e1b["rows"]
    om = np.array([r["omega"] for r in rows])
    ax.loglog(om, [r["coef_rmse_ls"] for r in rows], "o-", color="#c0392b", ms=2.5, lw=0.9, label="coef., LS")
    ax.loglog(om, [r["coef_rmse_nnls"] for r in rows], "s-", color="#e67e22", ms=2.5, lw=0.9, label="coef., NNLS")
    colors = ["#1f4e79", "#2e86c1", "#76b7e5"]
    for k in range(len(rows[0]["group_rmse_ls"])):
        lab = f"group {rows[0]['group_sizes'][k]}-col."
        ax.loglog(om, [r["group_rmse_ls"][k] for r in rows], "^-", color=colors[k], ms=2.2, lw=0.8, label=lab)
        ax.loglog(om, [r["group_rms_bound"][k] for r in rows], "--", color=colors[k], lw=0.6)
    ax.invert_xaxis()
    ax.set_xlabel(r"within-group spread $\xi$")
    ax.set_ylabel("RMS error")
    ax.set_title("(a) coefficient vs. group error")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, labelspacing=0.2, fontsize=4.6)


def panel_phase(ax, e1c):
    s = np.array(e1c["separations"])
    sig = np.array(e1c["sigmas"])
    rate = np.array(e1c["success_rate"])
    X, Y = np.meshgrid(sig, s)
    pc = ax.pcolormesh(X, Y, rate, shading="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(e1c["t2_boundary_sigma"], s, "r-", lw=0.9, label="T2 boundary")
    emp = [(e, si) for e, si in zip(e1c["largest_sigma_with_rate_ge_1_minus_delta"], s) if e is not None]
    if emp:
        ax.plot([e for e, _ in emp], [si for _, si in emp], "k.", ms=3, label=r"empirical $\geq0.95$")
    ax.set_xlabel(r"noise $\sigma$ (tolerance $\eta=1$)")
    ax.set_ylabel(r"separation $s_g$")
    ax.set_title(r"(b) $P(\|\mathrm{err}_g\|\leq\eta)$")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=False, fontsize=4.6, handletextpad=0.3)
    cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=4.2, length=1.5)
    cb.set_label("success rate", fontsize=4.6, labelpad=1)


def panel_recovery(ax, e1a):
    Js = [r["J"] for r in e1a]
    x = np.arange(len(Js))
    w = 0.26
    for i, (key, lab, col) in enumerate([("components", "matroid components", "#1f4e79"),
                                         ("bkw", "Belsley--Kuh--Welsch", "#7f8c8d"),
                                         ("pairwise", r"pairwise $\rho>0.99$", "#c0392b")]):
        vals = [r["recovery_rate"][key] for r in e1a]
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([str(J) for J in Js])
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"number of components $J$")
    ax.set_ylabel("")
    ax.set_title("(c) planted grouping recovered")
    ax.legend(loc="upper center", frameon=False, fontsize=4.4, handletextpad=0.3, labelspacing=0.15, ncol=1)


def panel_coverage(ax, e2):
    cov = e2["interval_coverage"]
    wid = e2["interval_width_median"]
    methods = [("posterior_correct", "posterior"), ("posterior_shifted", "shifted\nprior"),
               ("bootstrap", "bootstrap")]
    x = np.arange(len(methods))
    w = 0.36
    near = [cov[m]["near"] for m, _ in methods]
    sep = [cov[m]["separated"] for m, _ in methods]
    ax.bar(x - w / 2, near, w, color="#c0392b", label="near-dependent blocks")
    ax.bar(x + w / 2, sep, w, color="#1f4e79", label="separated components")
    for i, (m, _) in enumerate(methods):
        ax.text(x[i] - w / 2, near[i] + 0.02, f"{wid[m]['near']:.2f}", ha="center", fontsize=4.4)
        ax.text(x[i] + w / 2, sep[i] + 0.02, f"{wid[m]['separated']:.2f}", ha="center", fontsize=4.4)
    ax.axhline(0.95, color="k", lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in methods], fontsize=4.8)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 0.95])
    ax.set_ylabel("coverage")
    ax.set_title("(d) 95% intervals, one coef.")
    ax.legend(loc="upper center", frameon=False, fontsize=4.4, handletextpad=0.3, labelspacing=0.15)


def main():
    with open(RESULTS) as f:
        r = json.load(f)
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 1.95), gridspec_kw={"wspace": 0.62})
    panel_conditioning(axes[0], r["e1b"])
    panel_phase(axes[1], r["e1c"])
    panel_recovery(axes[2], r["e1a"])
    panel_coverage(axes[3], r["e2"])
    out = os.path.join(HERE, "fig_synthetic.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
