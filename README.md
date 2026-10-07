# Identifiability-Aware Source Attribution (IASA): supplementary code

Code for the submission *Identifiability-Aware Source Attribution of Air Pollution*.
It contains the IASA procedure (Algorithm 1 of the paper), the New Delhi experimental
platform, the controlled experiments and the observed-week analysis, the scripts that
produce the paper's tables and figures, and the unit tests.  Data files and result
files are not included (see **Data**).

## Installation

Python 3.12 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

All inverse computation uses PyTorch.  The transport operator is built in `float32`;
projection, diagnostics and fitting use `float64`.  Every script runs on CPU; pass
`--device cuda` to the experiment runner to use a GPU for the controlled experiments.

## Layout

| Path | Contents |
|---|---|
| `model/iasa/response.py` | Wind-driven Gaussian-puff transport operator \(A(\omega)\) (paper Section 3) |
| `model/iasa/background.py`, `projection.py` | Nuisance basis \(Q\) and the projection \(P_Q^\perp\) |
| `model/iasa/diagnostics.py` | Rank, \(\sigma_J\), visibility, weak set, absorption, coherence |
| `model/iasa/grouping.py` | Exact components \(\mathcal P^\star\), separation \(s_g\), reportable partitions at \(\tau_\theta\) |
| `model/iasa/merge.py` | Pairwise coherence merge of declared source blocks |
| `model/iasa/fit.py` | Nonnegative least squares by projected FISTA, residual adequacy test |
| `model/iasa/activity.py`, `wind.py` | Temporal basis functions; wind conversion and kernel interpolation onto the grid |
| `model/iasa/footprints.py`, `reporting.py` | Per-sensor contributions and footprints; report tables |
| `model/iasa/refine.py`, `fieldformer_adapter.py` | Optional refinement and learned wind imputer (not used in the reported runs) |
| `experiments/iasa_pol/` | New Delhi platform (`nd_platform.py`), Experiments 1--11 and the observed mode (`experiments.py`), runner (`run_experiment.py`), observed-week bounds and bootstrap (`signal_target_weeks.py`), summaries, configs |
| `experiments/iasa_synthetic/` | A synthetic study (not reported in the paper) |
| `baselines/` | Baselines of Experiment 11 (`receptor.py`: plain NNLS, transport-based mass balance, PMF/NMF) and a vendored inference-only FieldFormer wind model |
| `data/pol_weather.py` | Loader for the hourly station records |
| `sim/` | Proxy-inventory loader (`pol_sources.py`) and advection--diffusion utilities (`polsim.py`, used for the structural-mismatch case of Experiment 5) |
| `evaluation/` | `eval_pol_iasa.py` (report tables) and `iasa_pol/configs/` |
| `scripts/` | Sanity gates, runtime smoke check, wind imputation, and `figures/` (figure scripts) |
| `tests/` | Unit tests |
| `docs/iasa_workflow.md` | Pipeline steps and the artifact schema |

## Data

The data files are not part of this archive.  Place them in `sim/`:

- `govdata_1H_current.csv`: hourly regulatory records for New Delhi, 1 May 2018 to
  31 October 2020, with columns `monitor_id, timestamp_round, AT, RH, WD, WS, pm10,
  pm25`, from the public Central Pollution Control Board portal.
- `govdata_locations.csv`: station coordinates, with columns `Monitor ID, Latitude,
  Longitude, Location`.
- `brick_kilns_intensity_80x80.npy`, `industries_intensity_80x80.npy`,
  `population_density_intensity_80x80.npy` and
  `traffic_{00,06,12,18}_intensity_80x80.npy`: the proxy maps on an \(80\times80\)
  grid, built from the regional emissions inventory, the gridded population product
  and the road-network map cited in the paper.  `sim/pol_sources.py` crops each map
  to the \(40\times40\) study window and divides it by its own 99th percentile.

## Settings used in the paper

Lag \(L=12\) hours (except the lag sweep of Experiment 7); nuisance basis of rank 4
(constant, linear trend, one daily sine--cosine pair) except in the nuisance-basis
variations of Experiments 3 and 11; visibility threshold \(\tau_v=0\); separation
threshold \(\tau_\theta=\sqrt{1-0.99^2}=0.141\), which equals coherence \(0.99\) for an
operator with two columns; ridge weight \(\lambda=0\); \(\delta=0.05\); 1,000 bootstrap
refits.  These are set in the configs and the code defaults; nothing is tuned on
recovery error.

## Reproducing the results

Run from the repository root.

1. Tests and sanity gates:

   ```bash
   python -m pytest -p no:cacheprovider tests
   python scripts/run_iasa_sanity.py --gate all
   ```

2. Controlled Experiments 1--10 (one run per config, seed 0):

   ```bash
   for c in evaluation/iasa_pol/configs/exp*.json; do
     python experiments/iasa_pol/run_experiment.py --config "$c" --seed 0 \
       --out evaluation/iasa_pol/runs
   done
   ```

3. Experiment 11 (baselines, seeds 0--2; the result includes the group-signal scores
   of Table 1):

   ```bash
   for s in 0 1 2; do
     python experiments/iasa_pol/run_experiment.py \
       --config experiments/iasa_pol/configs/exp11.json --seed $s \
       --out evaluation/iasa_pol/runs_signal_target
   done
   ```

4. Observed weeks 1--4 (1--28 May 2018), then the error bounds of Theorem 5.9(c) and
   the parametric bootstrap of the shares:

   ```bash
   for k in 1 2 3 4; do
     python experiments/iasa_pol/run_experiment.py \
       --config evaluation/iasa_pol/configs/observed_week$k.json --seed 0 \
       --out evaluation/iasa_pol/runs/week$k
   done
   python experiments/iasa_pol/signal_target_weeks.py
   ```

   `signal_target_weeks.py` also estimates the noise level from the two co-located
   monitors and writes `evaluation/iasa_pol/signal_target/weeks.json`.

5. Tables and figures:

   ```bash
   python experiments/iasa_pol/summarize_results.py \
     --runs evaluation/iasa_pol/runs --summaries evaluation/iasa_pol/summaries
   python evaluation/eval_pol_iasa.py --runs evaluation/iasa_pol/runs \
     --out evaluation/iasa_pol/reports
   cd scripts/figures && python make_figures.py && python make_platform_figure.py
   ```

Each run writes `config.resolved.json` (configuration, seed, device, dtype and
library versions), `result.json` and `arrays.npz`; `docs/iasa_workflow.md` describes
the fields.

## Scope

The results are conditional on the declared wind field, transport model, proxy
inventories, temporal basis, lag and nuisance basis.  Reported shares are fractions
of the fitted projected concentration signal, not emission shares.

## Internal: cluster runtime (not in the supplementary archive)

### Container and SLURM commands

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

The supplementary archive is `paper_aistats/iasa_supplementary_code.zip` (code only,
anonymized; built from the tracked code directories with this README).
