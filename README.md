# SSMT – State-Space Multi-Taper Spectrogram
Python implementation of the method described in  
> S.-E. Kim, M. Behr, D. Ba, and E. N. Brown, *"State-Space Multitaper Time-Frequency Analysis,"* Proceedings of the National Academy of Sciences, vol. 115, no.1, pp. E5-E14, Jan 2018. (http://www.pnas.org/content/early/2017/12/15/1702877115) 

The numerical core is a line-for-line translation of the original MATLAB implementation released under CC BY-NC-SA 4.0 at <https://github.com/seekim/SSMT>.


## 1. Installation

```bash
git clone https://github.com/bowen-w-zheng/pyssmt.git
cd pyssmt
pip install -e .
```

## 2. Quick test to reproduce the spectrogram
```bash
ssmt examples/SED10.mat
```

## 3. Using SSMT on your own recording
```python
import numpy as np, matplotlib.pyplot as plt
from ssmt.tapers import dpss_tapers
from ssmt.em import estimate_parameters
from ssmt.spect import ss_mt
from ssmt.plot import show_spectrogram

# --- raw_signal: 1-D NumPy array (samples,) ---
raw_signal = np.load("my_signal.npy")
fs        = 1000         # sampling rate  (Hz)
win_sec   = 0.5          # window length  (s)
TW, K     = 3, 5         # DPSS parameters, with K < 2*TW - 1

NW   = int(win_sec * fs)
nwin = len(raw_signal) // NW
yy   = raw_signal[:NW*nwin].reshape(NW, nwin, order="F")   # column-major

# 1.  learn noise parameters on a short slice
GUESS_WINDOW_LENGTH = 200 # (Using 100 s / win_sec = 200 sample points)
freq_slice = np.fft.fft(
    dpss_tapers(NW, TW, K)[0][:,:,None] * yy[:,None,:GUESS_WINDOW_LENGTH], axis=0
)
pars = estimate_parameters(freq_slice, verbose=False)

# 2.  full spectrogram
spect, _ = ss_mt(yy, fs, TW, K,
                 pars["sn"], pars["on"], pars["is0"], pars["iv0"])

# 3.  plot
show_spectrogram(spect, win_sec, fs)
plt.show()
```
