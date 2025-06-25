from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import Tuple

@njit(fastmath=True, parallel=True, cache=True)
def _em_step(systemNoise, observationNoise,
                        initialState, initialVariance,
                        frequencyY, alpha, beta, obsnoise_cutoff) -> Tuple:
    nw, K, N = frequencyY.shape
    # ---------- work tensors -----------------------------------------
    xp  = np.empty_like(frequencyY)
    Pp  = np.empty((nw, K, N), np.float64)
    xs  = np.empty_like(frequencyY)
    Ps  = np.empty_like(Pp)
    xsm = np.empty_like(frequencyY)
    Psm = np.empty_like(Pp)
    lag = np.empty_like(Pp)

    x0_arr = np.empty((nw, K), frequencyY.dtype)
    P0_arr = np.empty((nw, K), np.float64)

    # ======================  FORWARD (Kalman) ========================
    for i in prange(nw):
        for k in range(K):
            # ----- patch: match reference initial-prediction semantics -----
            xp[i, k, 0] = 0.0 + 0.0j          # statePred[:, :, 0]  ≡ 0
            Pp[i, k, 0] = 0.0                 # variancePred[:, :, 0] ≡ 0
            xs[i, k, 0] = initialState[i, k]  # stateEst[:, :, 0]
            Ps[i, k, 0] = initialVariance[i, k]  # varianceEst[:, :, 0]
        for t in range(1, N):
            for k in range(K):
                xp[i, k, t] = xs[i, k, t-1]
                Pp[i, k, t] = Ps[i, k, t-1] + systemNoise[i, k]
                Kg          = Pp[i, k, t] / (Pp[i, k, t] + observationNoise[i, k])
                xs[i, k, t] = xp[i, k, t] + Kg * (frequencyY[i, k, t] - xp[i, k, t])
                Ps[i, k, t] = (1.0 - Kg) * Pp[i, k, t]

    # ======================  BACKWARD (RTS)  =========================
    for i in prange(nw):
        for k in range(K):
            xsm[i, k, N-1] = xs[i, k, N-1]
            Psm[i, k, N-1] = Ps[i, k, N-1]

        for t in range(N-2, -1, -1):
            for k in range(K):
                G = Ps[i, k, t] / Pp[i, k, t+1]
                diff          = xsm[i, k, t+1] - xp[i, k, t+1]
                xsm[i, k, t]  = xs[i, k, t] + G * diff
                Psm[i, k, t]  = Ps[i, k, t] + G*G * (Psm[i, k, t+1] - Pp[i, k, t+1])
                lag[i, k, t+1] = G * Psm[i, k, t+1]

        # smoothed k = 0
        for k in range(K):
            G0          = Ps[i, k, 0] / Pp[i, k, 1]
            x0_arr[i, k] = xp[i, k, 0] + G0 * (xsm[i, k, 0] - xp[i, k, 0])
            P0_arr[i, k] = Pp[i, k, 0] + G0*G0 * (Psm[i, k, 0] - Pp[i, k, 0])
            lag[i, k, 0] = G0 * Psm[i, k, 0]

    # ======================  M-STEP : Q  =============================
    sn_next = np.empty_like(systemNoise)
    for i in prange(nw):
        for k in range(K):
            # Match the reference exactly:
            # sys_state_term = 2 * (|x̂0|² + Σ(t=0 to N-2)|x̂t|²) + |x̂(N-1)|²
            # sys_var_term   = 2 * (P̂0 + Σ(t=0 to N-2)P̂t) + P̂(N-1)
            
            state_sum = 0.0
            var_sum = 0.0
            for t in range(N-1):  # t = 0 to N-2
                state_sum += np.abs(xsm[i, k, t])**2
                var_sum += Psm[i, k, t]
            
            sys_state_term = 2.0 * (np.abs(x0_arr[i, k])**2 + state_sum) + np.abs(xsm[i, k, N-1])**2
            sys_var_term = 2.0 * (P0_arr[i, k] + var_sum) + Psm[i, k, N-1]
            
            # Sample covariance: x̂0 * x̂0* + Σ(t=0 to N-2) x̂t * x̂(t+1)*
            sample_cov = x0_arr[i, k] * np.conj(xsm[i, k, 0])
            for t in range(N-1):
                sample_cov += xsm[i, k, t] * np.conj(xsm[i, k, t+1])
            
            # Lag covariance sum over all time points
            lag_cov_sum = 0.0
            for t in range(N):
                lag_cov_sum += lag[i, k, t]
            
            sn_next[i, k] = (sys_state_term + sys_var_term
                           - 2.0 * sample_cov.real - 2.0 * lag_cov_sum + beta) / (alpha + N)

    # ======================  M-STEP : R  =============================
    wnw = obsnoise_cutoff if obsnoise_cutoff is not None else nw
    numer_mat = np.zeros((wnw, K))

    for i in prange(wnw):
        for k in range(K):
            acc = 0.0
            for t in range(N):
                acc += ( np.abs(frequencyY[i, k, t])**2
                       + np.abs(xsm[i, k, t])**2
                       + Psm[i, k, t]
                       - 2.0*np.real(frequencyY[i, k, t] * np.conj(xsm[i, k, t])) )
            numer_mat[i, k] = acc

    numer_vec = numer_mat.sum(axis=0)
    denom     = N*wnw + alpha - 1.0
    r_vec     = (beta + numer_vec) / denom

    on_next = np.empty_like(observationNoise)
    for k in range(K):
        on_next[:, k] = r_vec[k]

    # ======================  LOG-LIKELIHOOD  =========================
    ll = 0.0
    for t in range(N):
        for i in range(nw):
            for k in range(K):
                denom_r = Pp[i, k, t] + observationNoise[i, k]
                resid   = frequencyY[i, k, t] - xp[i, k, t]
                ll += ( - (resid.real**2 + resid.imag**2) / denom_r
                        - np.log(denom_r) - np.log(np.pi) )

    return sn_next, on_next, initialState, initialVariance, ll

def estimate_parameters(freq_Y: np.ndarray,
                        n_iter: int = 500,
                        conv: float = 1e-5,
                        obs_cut: int | None = None,
                        alpha: float = 1.0,
                        beta: float = 1.0,
                        r0: float = 100.0,
                        verbose: bool = True) -> dict:
    """Run EM; return dict with keys sn,on,is,iv,logL."""
    nw, k, _ = freq_Y.shape
    sn = np.ones((nw, k))
    on = np.full((nw, k), r0)
    is0 = freq_Y[..., 0]
    iv0 = (is0 * is0.conj()).real
    d = np.inf
    ll_trace = []
    it = 0
    while d > conv and it < n_iter:
        sn_n, on_n, _, _, ll = _em_step(
            sn, on, is0, iv0, freq_Y, alpha, beta, obs_cut
        )
        d = np.abs(sn - sn_n) / np.abs(sn + sn_n) # relative change
        d = d.mean()
        sn, on = sn_n, on_n
        ll_trace.append(ll)
        it += 1
        if verbose:
            print(f"[EM] iter {it:3d}  Δ={d:.2e}  LL={ll:.2e}", end="\r")
    if verbose:
        print()
    return dict(sn=sn, on=on, is0=is0, iv0=iv0, logL=np.array(ll_trace))
