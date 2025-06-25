from typing import Tuple
import numpy as np
from scipy.signal.windows import dpss

def dpss_tapers(nw: int, tw: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """DPSS tapers with MATLAB sign convention."""
    tapers, conc = dpss(nw, tw, k, norm=2, return_ratios=True, sym = True)
    tapers = tapers.T                      # (nw, k)
    tapers[:, tapers[0] < 0] *= -1.0
    return tapers, conc
