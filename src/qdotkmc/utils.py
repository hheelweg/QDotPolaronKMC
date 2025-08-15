import numpy as np
import scipy.linalg as la
import pandas as pd
from typing import Callable, Dict, Any, Tuple


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


# get diffusivity from single msd trajectory and times array
def get_diffusivity(msd, times, dim, tail_frac=0.5):
    """
    Minimal single-curve diffusivity:
      - Fit MSD(t) ~ a + b t on the last `tail_frac` of points.
      - Return D = b/(2*dim) and its standard error.
    """
    t = np.asarray(times, float)
    y = np.asarray(msd,   float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if t.size < 3:
        raise ValueError("Not enough points to fit.")

    # pick tail
    q = max(0.0, min(1.0, 1.0 - tail_frac))
    tmin = np.quantile(t, q)
    sel = t >= tmin
    T, Y = t[sel], y[sel]
    if T.size < 3:
        # last resort: last 3 points
        T, Y = t[-3:], y[-3:]

    # linear least squares: Y ≈ a + b T
    A = np.vstack([T, np.ones_like(T)]).T
    coef, residuals, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    b, a = coef[0], coef[1]

    # slope stderr
    yfit = A @ coef
    dof = max(len(T) - 2, 1)
    ss_res = float(np.sum((Y - yfit)**2))
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    b_stderr = float(np.sqrt(cov[0, 0]))

    D = b / (2.0 * dim)
    D_stderr = b_stderr / (2.0 * dim)
    return D, D_stderr




# summary multiple diffusivities
def summarize_diffusivities(msds: np.ndarray,
                            times: np.ndarray,
                            dim: int,
                            *,
                            fit_single: Callable[..., Tuple[float, float, float, float, float, Dict[str, Any]]],
                            # pass-through options for the single-curve fitter:
                            slope_tol: float = 0.20,
                            min_points: int = 8,
                            fallback_quantiles: Tuple[float, float] = (0.5, 0.9)
                            ):
    msds = np.asarray(msds, dtype=float)
    times = np.asarray(times, dtype=float)
    if msds.ndim != 2:
        raise ValueError("msds must have shape (R, T).")
    R, T = msds.shape
    if times.shape != (T,):
        raise ValueError("times must have length T matching msds.shape[1].")

    D_list, sD_list, r2_list, winfo, used_fb = [], [], [], [], []

    # --- fit each realization ---
    for i in range(R):
        msd_i = msds[i]
        # Skip if too many NaNs
        if not np.isfinite(msd_i).any():
            continue
        try:
            D_i, b, a, b_se, r2, info = fit_single(
                msd_i, times, dim,
                slope_tol=slope_tol,
                min_points=min_points,
                fallback_quantiles=fallback_quantiles
            )
        except Exception:
            continue  # skip failed fits

        sigma_D_i = float(b_se) / (2.0 * dim)
        if not (np.isfinite(D_i) and np.isfinite(sigma_D_i) and sigma_D_i >= 0):
            continue

        D_list.append(float(D_i))
        sD_list.append(float(sigma_D_i))
        r2_list.append(float(r2))
        winfo.append((float(info.get('tmin', np.nan)),
                      float(info.get('tmax', np.nan)),
                      int(info.get('npts', 0))))
        used_fb.append(bool(info.get('used_fallback', False)))

    if len(D_list) == 0:
        raise ValueError("No realization could be fitted to extract diffusivity.")

    D_arr = np.asarray(D_list)
    sD_arr = np.asarray(sD_list)

    # Inverse-variance weights; guard tiny/zero errors
    w = 1.0 / np.maximum(sD_arr, 1e-300)**2
    W = np.sum(w)

    D_weighted = float(np.sum(w * D_arr) / W)
    D_weighted_err = float(1.0 / np.sqrt(W))  # internal (within-fit) SE

    # Between-realization variability (captures disorder spread)
    D_weighted_sem = float(D_arr.std(ddof=1) / np.sqrt(D_arr.size)) if D_arr.size > 1 else np.nan

    # Total uncertainty (recommended): combine internal + between-realization in quadrature
    D_weighted_total = float(np.sqrt(D_weighted_err**2 + (0.0 if np.isnan(D_weighted_sem) else D_weighted_sem**2)))

    # --- pooled MSD fit (cross-check) ---
    msd_pooled = np.nanmean(msds, axis=0)
    D_pooled, b, a, b_se, r2_pooled, info_pooled = fit_single(
        msd_pooled, times, dim,
        slope_tol=slope_tol,
        min_points=min_points,
        fallback_quantiles=fallback_quantiles
    )
    D_pooled_err = float(b_se) / (2.0 * dim)

    # Package per-realization details
    per_real = np.zeros(len(D_arr), dtype=[
        ('D', float), ('sigma_D', float), ('r2', float),
        ('tmin', float), ('tmax', float), ('npts', int), ('used_fallback', bool)
    ])
    per_real['D'] = D_arr
    per_real['sigma_D'] = sD_arr
    per_real['r2'] = np.asarray(r2_list, dtype=float)
    if winfo:
        per_real['tmin'] = np.asarray([w[0] for w in winfo], dtype=float)
        per_real['tmax'] = np.asarray([w[1] for w in winfo], dtype=float)
        per_real['npts'] = np.asarray([w[2] for w in winfo], dtype=int)
    per_real['used_fallback'] = np.asarray(used_fb, dtype=bool)

    return dict(
        D_weighted=D_weighted,
        D_weighted_err=D_weighted_err,
        D_weighted_sem=D_weighted_sem,
        D_weighted_total=D_weighted_total,
        D_pooled=float(D_pooled),
        D_pooled_err=D_pooled_err,
        pooled_info=dict(info_pooled, r2=r2_pooled),
        per_real=per_real,
    )
