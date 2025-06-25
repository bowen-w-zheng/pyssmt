import numpy as np
from .tapers import dpss_tapers
"""
Generalised multitaper spectral estimation.

"""
def ss_mt(yy: np.ndarray, fs: int, tw: float, k: int,
          sn: np.ndarray, on: np.ndarray,
          is0: np.ndarray, iv0: np.ndarray) -> tuple[np.ndarray, dict]:
    nw, nwin = yy.shape
    tapers, _ = dpss_tapers(nw, tw, k,)
    mtY = tapers[:, :, None] * yy[:, None, :]
    freq_Y = np.fft.fft(mtY, axis=0)

    state = np.empty_like(freq_Y)
    var   = np.empty((nw, k, nwin))
    pred_s = np.empty_like(freq_Y)
    pred_v = np.empty_like(var)
    kgain  = np.empty_like(var)

    state[:, :, 0], var[:, :, 0] = is0, iv0
    for n in range(1, nwin):
        pred_s[:, :, n] = state[:, :, n-1]
        pred_v[:, :, n] = var[:, :, n-1] + sn
        kgain[:, :, n]  = pred_v[:, :, n] / (pred_v[:, :, n] + on)
        resid = freq_Y[:, :, n] - pred_s[:, :, n]
        state[:, :, n] = pred_s[:, :, n] + kgain[:, :, n] * resid
        var[:, :, n]   = (1.0 - kgain[:, :, n]) * pred_v[:, :, n]

    mt_spect = (np.abs(state)**2) / fs      # (nw,k,nwin)
    spect    = mt_spect.mean(axis=1)        # taper-avg
    return spect, dict(state=state, var=var, kgain=kgain, 
                       state_prediction = pred_s, variance_prediction = pred_v, resid = resid,
                       system_noise = sn, obs_noise = on, 
                       mtSpects = mt_spect)
