# stamp

STAMP is currently a container-first research codebase. The pollution workflow
depends on scientific Python packages that are available in the repository's
Singularity/Apptainer image, not in the host Python environment.

## Runtime

Use the existing image and overlay from the repository root:

- Image: `cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`
- Overlay: `overlay-25GB-500K.ext3`
- Apptainer/Singularity binary: `/share/apps/apptainer/bin/singularity`

The supported runtime command pattern is:

```bash
/share/apps/apptainer/bin/singularity exec --fakeroot \
  --overlay overlay-25GB-500K.ext3:ro \
  cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh && cd /scratch/ab9738/stamp && python3 scripts/smoke_iasa_runtime.py"
```

The smoke check verifies the runtime packages used by the pollution path:
`numpy`, `torch`, `pandas`, `pykrige`, and optional `scipy`. It also imports
`sim.polsim`, loads the pollution source maps, builds a 40x40 grid, and reports
sensor metadata shapes without running a simulation.

## New Delhi Wind Imputation

Task 2 adds a FieldFormer ImputeFormer wind pipeline. The implementation copies
the compact fixed-node ImputeFormer model into this repository and trains a
fresh checkpoint on the local government wind observations.

For a quick smoke run, use a short time window and a tiny training budget:

```bash
/share/apps/apptainer/bin/singularity exec --fakeroot \
  --overlay overlay-25GB-500K.ext3:ro \
  cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh && cd /scratch/ab9738/stamp && python3 scripts/impute_new_delhi_wind.py --start '2018-05-01 00:00:00+05:30' --end '2018-05-02 23:00:00+05:30' --epochs 1 --windows 24 --window-stride 24 --batch-size 2 --val-batch-size 2 --device cpu --output /tmp/new_delhi_wind_smoke.npz --checkpoint /tmp/imputeformer_wind_smoke.pt"
```

The full-range default output is `data/new_delhi_wind_imputed.npz`; run it on a
short SLURM GPU allocation if the login node is unsuitable.

Host Python is not currently a supported runtime for this repository. In the
current environment, host Python does not provide the required scientific stack.

If container startup or imports are not suitable on the login node, run the same
smoke script through a short SLURM allocation/job using the existing
GPU/container environment. The smoke script itself does not require GPU compute.
