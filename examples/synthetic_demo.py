#!/usr/bin/env python
"""
examples/synthetic_demo.py
Compare periodogram, multi-taper, and state-space multi-taper on a
synthetic chirp with known ground-truth.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ssmt.tapers import dpss_tapers
from ssmt.em     import estimate_parameters
from ssmt.spect  import ss_mt

# ------------------------------------------------------------------
# 1)  Simulate a 5-min linear chirp (8 → 18 Hz) + white noise
# ------------------------------------------------------------------
fs      = 250       # Hz
T_sec   = 300       # 5 min
N       = fs * T_sec
t       = np.arange(N) / fs

f0, f1  = 8.0, 18.0
phase   = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T_sec * t**2)
signal  = np.sin(phase)
y       = signal + 0.1 * np.random.randn(N)

# ------------------------------------------------------------------
# 2)  Common windowing
# ------------------------------------------------------------------
win_sec = 2
NW      = win_sec * fs                  # 500 samples / window
nwin    = N // NW
yy      = y[:nwin*NW].reshape(NW, nwin, order="F")

TW, K   = 2, 3
tapers, _ = dpss_tapers(NW, TW, K)

# frequency axis for plotting
f = np.arange(0, fs/2 + fs/NW, fs/NW)

# number of rows to display (0-20 Hz)
rows = int(20 * win_sec) + 1            # 41 rows

# ------------------------------------------------------------------
# 3)  Periodogram (rectangular window)
# ------------------------------------------------------------------
rect = np.ones((NW, 1)) / np.sqrt(NW)
per  = np.abs(np.fft.fft(rect * yy, axis=0))**2 / fs   # (NW, nwin)

# ------------------------------------------------------------------
# 4)  Classical multi-taper
# ------------------------------------------------------------------
mtY     = tapers[:, :, None] * yy[:, None, :]
mt_power = np.abs(np.fft.fft(mtY, axis=0))**2 / fs
mtspec  = mt_power.mean(axis=1)                         # (NW, nwin)

# ------------------------------------------------------------------
# 5)  SS-MT  – learn parameters on first 100 windows
# ------------------------------------------------------------------
freq_slice = np.fft.fft(tapers[:, :, None] * yy[:, None, :100], axis=0)
pars = estimate_parameters(freq_slice, verbose=False)

ssmt_spec, _ = ss_mt(
    yy, fs, TW, K,
    pars["sn"], pars["on"], pars["is0"], pars["iv0"]
)

# ------------------------------------------------------------------
# 6)  Prepare ground-truth ridge image
# ------------------------------------------------------------------
truth = np.zeros_like(per)                           # (NW, nwin)
inst_freq = f0 + (f1 - f0) * np.linspace(0, 1, nwin) # Hz
row_idx = np.round(inst_freq * win_sec).astype(int)  # bin index
truth[row_idx, np.arange(nwin)] = 1.0

# ------------------------------------------------------------------
# 7)  Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(
    4, 1, figsize=(9, 10), sharex=True, sharey=True,
    gridspec_kw=dict(left=0.08, right=0.9, hspace=0.25)
)

panels = [
    (truth,      "Ground-truth chirp (8 → 18 Hz)",    "Greys"),
    (per,        "Rectangular periodogram",           "jet"),
    (mtspec,     "Classical multi-taper",             "jet"),
    (ssmt_spec,  "State-space multi-taper (SS-MT)",   "jet"),
]

for a, (S, title, cmap) in zip(ax, panels):
    im = a.imshow(10*np.log10(S[:rows]),
                  origin="lower", aspect="auto", cmap=cmap,
                  vmin=-40, vmax=10,
                  extent=[win_sec/60, nwin*win_sec/60, 0, 20])
    a.set_title(title)
    a.set_ylabel("Frequency [Hz]")

ax[-1].set_xlabel("Time [min]")

# single shared colour-bar outside the stack
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])          # [left, bottom, w, h]
fig.colorbar(im, cax=cax, label="Power [dB]")

plt.tight_layout(rect=(0, 0, 0.9, 1))                # leave space for c-bar

plt.show()
