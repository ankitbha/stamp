# Task 10 — Controlled experiment suite (`iasa_pol`)

Ten one-factor controlled experiments plus an observed-New-Delhi study mode on a
single shared base platform, for identifiability-aware source apportionment.

## Layout

| Path | Role | Committed? |
|------|------|-----------|
| `nd_platform.py` | Shared New Delhi base platform (real inventory downsampled, regulatory sensor geometry, wind/basis/background/coefficient factories) | yes |
| `edge_hold_pde.py` | Experiment 5b **structural** generator (two-stage Heun on `sim/polsim.py` edge-hold primitives, `k=3e-4`), labeled `edge_hold_pde` | yes |
| `experiments.py` | Experiments 1–10 + observed mode; each returns an `accuracy` block and a `diagnostics` block | yes |
| `run_experiment.py` | sbatch-friendly runner for one configured experiment; writes provenance + artifacts | yes |
| `summarize_results.py` | Rolls `runs/` up into `summaries/` CSV + `all_experiments.json` | yes |
| `configs/*.json` | One config per experiment (declared `Q`/lag/mask/inventory version) | yes |
| `summaries/` | Small roll-up tables Task 11 reads | yes (dir) |
| `runs/` | Per-run heavy artifacts (`arrays.npz`, `result.json`, `config.resolved.json`) | **no** (git-ignored; regenerable) |

## Running (GPU node, via sbatch)

`run_experiment.py` and the suite use torch and must run on a SLURM GPU node
inside the container/overlay, never the login node.

```bash
# One experiment, reproducible from config + seed:
python3 experiments/iasa_pol/run_experiment.py \
    --config experiments/iasa_pol/configs/exp01.json --seed 0 --device cuda

# All experiments (loop configs), then summarize into committed tables:
for c in experiments/iasa_pol/configs/*.json; do
  python3 experiments/iasa_pol/run_experiment.py --config "$c" --seed 0 --device cuda
done
python3 experiments/iasa_pol/summarize_results.py
```

Each run writes `runs/<experiment>_seed<N>/` with `config.resolved.json`
(config + seed + git SHA + device/dtype + torch/cuda versions), `result.json`,
and `arrays.npz`. Everything is reproducible from the committed config + seed;
the primary `Q`, lag rule, fixed-zero mask, and inventory version are recorded in
provenance and are never selected from fit quality.

## Fast CI-style check

```bash
python3 scripts/run_iasa_sanity.py --gate experiments   # reduced-scale sweep of all 11
```

Included in `--gate all --strict-all`. Unit-tested in
`tests/test_iasa_experiments.py`.
