from typing import Dict
import numpy as np

def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    out = {k: data[k] for k in data.files}
    return out