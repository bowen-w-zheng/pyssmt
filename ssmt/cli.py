import argparse, sys, pathlib as pl
import numpy as np
from .dataio import load_sed10
from .tapers import dpss_tapers
from .em import estimate_parameters
from .spect import ss_mt
from .plot import show_spectrogram
import matplotlib.pyplot as plt

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m pyssmt",
                                description="State-Space Multitaper Spectrogram")
    p.add_argument("mat_file", type=pl.Path, help="SED10.mat path")
    p.add_argument("--tw", type=float, default=2, help="time-bandwidth")
    p.add_argument("-k",  type=int,   default=3, help="# tapers")
    args = p.parse_args(argv)

    fs = 250
    win_sec = 2


    y = load_sed10(args.mat_file, fs)
    nw     = int(win_sec * fs)
    N   = len(y) // nw # number of windows
    yy     = y[:N*nw].reshape((nw, N), order="F")

    tapers, _ = dpss_tapers(nw, args.tw, args.k)
    y_ex = np.broadcast_to(yy[:, None, :], (nw, args.k, N)).copy() 
    mtY = tapers[:, :, None] * y_ex 
    mtFrequencyY = np.fft.fft(mtY, n=nw, axis=0)  # (nw, k, N)

    obs_cut = 25*win_sec # 25 Hz
    guess_window = 150
    pars   = estimate_parameters(mtFrequencyY[:,:,:guess_window], obs_cut=obs_cut, verbose=True)

    spect, _ = ss_mt(yy, fs, args.tw, args.k,
                     pars["sn"], pars["on"], pars["is0"], pars["iv0"])
    show_spectrogram(spect, win_sec, fs)
    plt.show()

if __name__ == "__main__":
    main()
