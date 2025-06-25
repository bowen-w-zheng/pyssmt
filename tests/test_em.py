# tests/test_em.py
import numpy as np, ssmt
def test_em_shapes():
    nw, k, n = 10, 2, 5
    fake = (np.random.randn(nw,k,n) + 1j*np.random.randn(nw,k,n))
    pars = ssmt.em.estimate_parameters(fake, n_iter=2, verbose=False)
    assert pars["sn"].shape == (nw, k)
