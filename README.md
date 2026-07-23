# stamp

STAMP is a container-first research codebase. The active workflow is
**IASA** (identifiability-aware source apportionment) for city-scale
advection--diffusion systems, instantiated on a New Delhi PM\(_{2.5}\) platform.
The scientific Python stack lives in the repository's Singularity/Apptainer image,
not in the host Python environment.

For the full step-by-step pipeline, command reference, and artifact/provenance
schema, see [`docs/iasa_workflow.md`](docs/iasa_workflow.md).

## Runtime environment

- Image: `cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`
- Overlay: `overlay-25GB-500K.ext3` (provides torch via `source /ext3/env.sh`)
- Apptainer/Singularity: `/share/apps/apptainer/bin/singularity`
- Numerics: response construction uses `float32`; projection, diagnostics,
  fitting, covariance, and ensembles use `float64` on an explicit device
  (`cpu` default, `cuda` supported). `pandas`/`NumPy` are used only for
  CSV/NPZ ingestion and serialization; all inverse computation is PyTorch.
- SciPy is **not** a required solver dependency.

Base command pattern (CPU/login-node work such as smoke checks and the paper build):

```bash
/share/apps/apptainer/bin/singularity exec --fakeroot \
  --overlay overlay-25GB-500K.ext3:ro \
  cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh && cd /scratch/ab9738/stamp && <command>"
```

GPU work (paper-scale experiments, gates) runs through SLURM, not the login node:

```bash
sbatch --account=torch_pr_633_general --partition=l40s_public \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=24:00:00 <job.sh>
```

Host Python is not a supported runtime; it lacks the required scientific stack.

## Quickstart: minimal IASA run

Runtime smoke check (imports, source maps, 40x40 grid, sensor metadata; no simulation):

```bash
... /bin/bash -lc "source /ext3/env.sh && cd /scratch/ab9738/stamp && python3 scripts/smoke_iasa_runtime.py"
```

Minimal end-to-end sanity gate (response -> projection -> diagnostics -> fit ->
merge on a tiny synthetic platform):

```bash
... python3 scripts/run_iasa_sanity.py --gate end_to_end
```

Run one controlled experiment and one observed window at paper resolution
(GPU; see `docs/iasa_workflow.md` for the full sweep and SLURM arrays):

```bash
... python3 experiments/iasa_pol/run_experiment.py \
      --config evaluation/iasa_pol/configs/exp01.json --device cuda \
      --out evaluation/iasa_pol/runs
```

## Sanity gates

`scripts/run_iasa_sanity.py --gate <name>` runs deterministic gates from public
APIs. Gates: `task3a`, `response`, `projection`, `parity`, `diagnostics`, `fit`,
`merge`, `end_to_end`, `wind_field`, `footprints`, `refine`, `fieldformer_train`,
`calibration` (S7), `experiments` (Task 10 sweep), `reporting` (Task 11), and
`all`. `--gate all` runs the light regression set; `--gate all --strict-all` adds
the heavier calibration, experiment-sweep, and reporting gates.

## Evaluation and reporting

- Controlled sweep + observed windows: `experiments/iasa_pol/run_experiment.py`
  over `evaluation/iasa_pol/configs/*.json` (controlled `expNN`, observed
  `observed_weekK`).
- Roll-ups: `experiments/iasa_pol/summarize_results.py` (controlled) and
  `summarize_weeks.py` (observed weeks, Tier 0).
- Paper tables: `evaluation/eval_pol_iasa.py --runs evaluation/iasa_pol/runs
  --out evaluation/iasa_pol/reports` emits `report.{json,md}` and per-table CSVs.

## New Delhi wind imputation

Observed `WD/WS` are converted to transport vectors and completed on the response
grid by a **kernel coordinate-query imputer** (`model/iasa/wind.py`,
`KernelCoordinateQueryImputer`), the adopted default. A learned FieldFormer
coordinate-query model was trained and evaluated
(`scripts/train_fieldformer_wind.py`) but **not adopted**: on held-out station
vectors it beat the kernel interpolator (RMSE 1.12 vs 1.55) but lost to a
non-spatial city-mean baseline (1.06), so the spatially resolved kernel field is
used. Saved wind products keep valid/observed `*_mask` fields and explicit
`*_missing_mask`, `vector_mask`, and `mask_convention` metadata so downstream code
never guesses mask polarity.

## Interpretation caveats

- Identifiability certificates are conditional on the declared inventory,
  transport, temporal basis, lag, background basis, observation mask, and noise
  assumptions.
- A calibrated residual rejection shows model inadequacy but **not its cause**;
  non-rejection does **not** establish inventory completeness.
- Report groups (connected components) are conservative, deterministic merges,
  not a guaranteed finest partition.
- Inventory-scenario rows are robustness comparisons, never confidence-interval
  draws; transport uncertainty and inventory robustness occupy separate fields.
- Reported percentages are fractions of fitted inventory-attributed sensor signal,
  not physical-emission shares.

## Paper

Sources are in `paper/` (AAAI 2027 format; the `aaai2027.sty`/`aaai2027.bst` style
files are vendored in `paper/`). The compiled `paper/main.pdf` is committed and is
the reference build. Rebuild in a full TeX Live that provides the AAAI-required
`newtx` fonts (`newtxtext`/`newtxmath`), which the repository container image does
**not** include:

```bash
cd paper && pdflatex -interaction=nonstopmode main.tex && \
  bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Legacy code

Archived legacy files live only in the git-ignored `archive/` directory if kept
locally. They are **out of the active repository contract**: active code must not
depend on them, and they are not present in clean checkouts. No heat/SWE, SimGrad,
free-field recovery, or old pollution-calibration workflow is maintained.
