import numpy as np
import scipy.linalg as la
import pandas as pd


"""
This file contains some helper functions that will be called
"""


# diagonalize Hamiltonian
# NOTE : might want to make this more efficient with GPU/torch etc.
def diagonalize(H, S=None):
    """
    Diagonalize a real, symmetrix matrix and return sorted results.
    
    Return the eigenvalues and eigenvectors (column matrix) 
    sorted from lowest to highest eigenvalue.
    """
    E,C = la.eigh(H,S)
    E = np.real(E)
    #C = np.real(C)

    idx = E.argsort()
    #idx = (-E).argsort()
    E = E[idx]
    C = C[:,idx]

    return E,C


def export_msds(times, msds, file_name = "msds.csv"):

    # obtain mean MSD averaged over all realizations of noise
    msds_mean = np.mean(msds, axis = 0)

    # create df and save it
    data = np.column_stack([times, msds_mean.T, msds.T])
    columns = ["time"] + ["ave. msd"] + [f"msd lattice_{i}" for i in range(msds.shape[0])]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_name, index=False)


def get_diffusivity(msd, times, dim, *,
                    slope_tol=0.20,     # tolerance around slope=1 in log-log
                    min_points=8,       # minimum points in the detected window
                    fallback_quantiles=(0.5, 0.9)):  # if auto-window fails
    """
    Estimate diffusivity D from MSD(t) using Einstein relation:
        MSD(t) ~ a + b t  =>  D = b / (2*dim)
    Strategy:
      1) Find the longest log-log region where slope ~ 1 (diffusive).
      2) Fit MSD ~ a + b t in *linear* space on that window.
      3) If no region long enough, fall back to fitting the tail
         between the given quantiles of time.

    Returns
    -------
    D, b, a, b_stderr, r2, info
      info: dict with keys:
        'used_fallback' (bool), 'tmin','tmax','npts','slope_mean'
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(msd,   dtype=float)

    # Basic sanity masks
    mask = np.isfinite(t) & np.isfinite(y)
    mask &= t > 0.0
    mask &= y > 0.0
    if mask.sum() < 3:
        raise ValueError("Not enough valid (positive, finite) points to estimate D.")

    t = t[mask]; y = y[mask]

    # 1) Local log-log slope (central differences where possible)
    logt = np.log(t)
    logy = np.log(y)
    # central gradient on interior; forward/backward at ends
    slope = np.gradient(logy, logt)

    # Find longest consecutive block where |slope-1| < slope_tol
    in_diff = np.abs(slope - 1.0) < slope_tol
    best_start = best_len = 0
    cur_start = cur_len = 0
    for i, ok in enumerate(in_diff):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    used_fallback = False
    if best_len >= min_points:
        fit_slice = slice(best_start, best_start + best_len)
    else:
        # 2) Fallback: fit the tail between given quantiles
        q0, q1 = fallback_quantiles
        if not (0.0 <= q0 < q1 <= 1.0):
            raise ValueError("fallback_quantiles must be within [0,1] and q0 < q1.")
        tmin = np.quantile(t, q0)
        tmax = np.quantile(t, q1)
        fit_slice = (t >= tmin) & (t <= tmax)
        if fit_slice.sum() < max(3, min_points//2):
            # last resort: take the last max(min_points, 5) points
            k = max(min_points, 5)
            fit_slice = slice(max(0, t.size - k), t.size)
        used_fallback = True

    T = t[fit_slice]; Y = y[fit_slice]
    if T.size < 3:
        raise ValueError("Diffusive regime too short even after fallback.")

    # 3) Linear least squares: MSD ~ a + b t
    A = np.vstack([T, np.ones_like(T)]).T
    coef, residuals, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    b, a = coef[0], coef[1]
    yfit = A @ coef

    ss_res = np.sum((Y - yfit)**2)
    ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    dof = max(len(T) - 2, 1)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    b_stderr = float(np.sqrt(cov[0, 0]))

    D = b / (2.0 * dim)

    # Diagnostics
    if isinstance(fit_slice, slice):
        i0, i1 = fit_slice.start or 0, fit_slice.stop or T.size
        slope_mean = float(np.mean(slope[i0:i1]))
        tmin, tmax = float(t[i0]), float(t[i1-1])
        npts = int(i1 - i0)
    else:
        slope_mean = float(np.mean(slope[fit_slice]))
        tmin, tmax = float(T[0]), float(T[-1])
        npts = int(T.size)

    # get standard error
    sigma_D = b_stderr / (2.0 * dim)

    return D, b, a, b_stderr, r2, sigma_D

