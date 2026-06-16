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

Host Python is not currently a supported runtime for this repository. In the
current environment, host Python does not provide the required scientific stack.

If container startup or imports are not suitable on the login node, run the same
smoke script through a short SLURM allocation/job using the existing
GPU/container environment. The smoke script itself does not require GPU compute.
