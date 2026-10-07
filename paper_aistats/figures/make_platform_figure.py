"""Platform figure: the four proxy source maps on the 40x40 grid with the regulatory
sensor cells, read through the same loaders the experiments use."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from experiments.iasa_pol import nd_platform as P  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})


def main():
    names, maps, _ = P.load_inventory_maps((40, 40))
    # Traffic is one road-network source: the mean of the time-slot maps, as in
    # nd_platform.four_group_inventory.
    traffic = maps[[i for i, n in enumerate(names) if n.startswith("traffic_")]].mean(axis=0)
    groups = [("brick kilns", maps[names.index("brick_kilns")]),
              ("industries", maps[names.index("industries")]),
              ("population", maps[names.index("population_density")]),
              ("traffic", traffic)]
    xy, _, _ = P._regulatory_grid_cells((40, 40))
    fig, axes = plt.subplots(1, 4, figsize=(3.35, 1.05),
                             gridspec_kw=dict(wspace=0.08, left=0.01, right=0.99, top=0.88, bottom=0.02))
    for ax, (title, m) in zip(axes.ravel(), groups):
        ax.imshow(np.clip(m, 0, 1).T, origin="lower", cmap="magma", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.scatter(xy[:, 0], xy[:, 1], s=3, c="#2ecc71", edgecolors="k", linewidths=0.25)
        ax.set_title(title, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(HERE, "fig_platform.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
