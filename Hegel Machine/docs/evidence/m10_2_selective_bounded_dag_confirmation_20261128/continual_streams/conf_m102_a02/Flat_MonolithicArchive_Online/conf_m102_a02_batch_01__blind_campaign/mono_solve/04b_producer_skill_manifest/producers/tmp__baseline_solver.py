"""Robust coordinate calibration and a two-mode relaxation model."""
import numpy as np
from scipy.optimize import least_squares


def _calibration(panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    # Median pairwise slopes initialize the fit without giving outliers leverage.
    a, b = np.triu_indices(len(u), 1)
    du = u[a] - u[b]
    dv = v[a] - v[b]
    slope = np.nanmedian(np.divide(dv, du, out=np.full_like(du, np.nan), where=abs(du)>1e-10), axis=0)
    shift = np.median(v - u*slope, axis=0)
    for j in range(u.shape[1]):
        r = v[:,j] - u[:,j]*slope[j] - shift[j]
        keep = abs(r) <= max(1e-7, 6*np.median(abs(r)))
        if keep.sum() >= 3:
            shift[j], slope[j] = np.linalg.lstsq(np.c_[np.ones(keep.sum()), u[keep,j]], v[keep,j], rcond=None)[0]
    return shift, slope


def _time_basis(t, z):
    def basis(r):
        return -np.expm1(-t[:,None]*r)
    def resid(r):
        B = basis(r)
        return (np.linalg.lstsq(B,z.T,rcond=None)[0].T @ B.T - z).ravel()
    r = least_squares(resid, [1.8,.4], bounds=([.8,.03],[8.,.79])).x
    return basis(r)


def _design(x):
    return np.c_[np.ones(len(x)), x, np.tanh(1.25*x)]


def solve_batch(study: dict) -> dict:
    tr = study['train']
    shift, slope = _calibration(study['panel_a'])
    A = _design((tr['x']-shift)/slope)
    Q = _design((study['test']['x']-shift)/slope)
    B = _time_basis(np.asarray(study['eval_times']), np.asarray(tr['traj']))
    y = Q @ np.linalg.lstsq(A,tr['y'],rcond=None)[0]
    amp = np.linalg.lstsq(B,tr['traj'].T,rcond=None)[0].T
    z = (Q @ np.linalg.lstsq(A,amp,rcond=None)[0]) @ B.T
    return {'y_hat':y, 'state_pred':z, 'diagnostics_vec':np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()])}
