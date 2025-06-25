from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.io as sio

def load_sed10(mat_path: Path, fs: int = 250, start_s: int = 600,
               stop_s: int = 1880, channel: int = 0) -> np.ndarray:
    """Return 1-D float64 trace (shape: samples,)."""
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = np.asarray(mat["data"], dtype=np.float64)
    if data.shape[0] > data.shape[1]:   # MATLAB (samples×chan)
        data = data.T
    y = data[channel, start_s*fs : stop_s*fs]
    return y
