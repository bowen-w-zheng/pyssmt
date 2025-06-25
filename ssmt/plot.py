from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def show_spectrogram(spect: np.ndarray, win_sec: float,
                     fs: int, fmax: int = 30,
                     vmin: int = -15, vmax: int = 10,
                     ax: plt.Axes | None = None) -> plt.Axes:
    rows = int(fmax * win_sec) + 1
    nwin = spect.shape[1]
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(10*np.log10(spect[:rows]),
                   origin="lower", aspect="auto", cmap="jet",
                   vmin=vmin, vmax=vmax,
                   extent=[win_sec/60, nwin*win_sec/60, 0, fmax])
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Frequency [Hz]")
    plt.colorbar(im, ax=ax, label="Power [dB]")
    return ax
