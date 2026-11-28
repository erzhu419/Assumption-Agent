"""Calibrated rational readout and affine state forecasts."""
import numpy as np
from scipy.optimize import least_squares


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a, b = np.triu_indices(len(u), 1)
    du = u[a] - u[b]
    slopes = np.nanmedian(np.divide(v[a] - v[b], du, out=np.full_like(du, np.nan), where=abs(du) > 1e-10), axis=0)
    shifts = np.median(v - u * slopes, axis=0)
    for k in range(u.shape[1]):
        r = v[:, k] - shifts[k] - slopes[k] * u[:, k]
        ok = abs(r) <= max(1e-7, 6 * np.median(abs(r)))
        if ok.sum() >= 3:
            shifts[k], slopes[k] = np.linalg.lstsq(np.c_[np.ones(ok.sum()), u[ok, k]], v[ok, k], rcond=None)[0]
    return shifts, slopes


def _aff(z):
    return np.c_[np.ones(len(z)), z]


def _quad(z):
    i, j = np.triu_indices(z.shape[1])
    return np.c_[_aff(z), z[:, i] * z[:, j]]


def _rational_readout(z, q, y):
    P, Q = _quad(z), _quad(q)
    keep = np.ones(P.shape[1], dtype=bool)
    keep[12] = False  # In the rational branch, w**2*g = -w**2-g/alpha.
    # Shrink mainly the quadratic correction, not the base affine law.
    R = np.diag(np.sqrt(0.002 * np.r_[np.zeros(5), np.full(10, .05),
                                     np.full(5, .1), np.ones(9)]))
    target = np.r_[y, np.zeros(len(R))]
    bounds = ([.025, .05, -.8, -.8, -.8], [5., 2., .8, .8, .8])
    starts = [(1., .42, .03, -.1, -.08), (1., .42, -.24, -.1, -.08), (.6, .8, .18, 0., 0.), (1.2, .4, .05, -.16, -.13)]

    def branch(root):
        def design(x, base, p):
            alpha, k, delta, b, d = p
            w = x[:, 2] - k*x[:, 0] + delta + b*x[:, 1] + d*x[:, 3]
            t = alpha * w**2
            g = 1/np.sqrt(1+t)-1 if root else -t/(1+t)
            return np.c_[base, base[:, keep] * g[:, None]]

        def fit(p):
            F = np.r_[design(z, P, p), R]
            c = np.linalg.lstsq(F, target, rcond=1e-11)[0]
            return F @ c - target, c

        opts = [least_squares(lambda p: fit(p)[0], p0, bounds=bounds,
                              max_nfev=110) for p0 in starts]
        p = min(opts, key=lambda result: result.cost).x
        return design(q, Q, p) @ fit(p)[1]

    # The rational quotient is primary; a weaker tail shape reduces
    # sensitivity to the fitted denominator on high-leverage inputs.
    return branch(True)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    shift, scale = _calibrate(study['panel_a'])
    z = (np.asarray(tr['x']) - shift) / scale
    q = (np.asarray(study['test']['x']) - shift) / scale
    y = _rational_readout(z, q, np.asarray(tr['y']))
    S = _aff(q) @ np.linalg.lstsq(_aff(z), np.asarray(tr['traj']), rcond=None)[0]
    return {'y_hat': y, 'state_pred': S,
            'diagnostics_vec': np.array([y.mean(), y.std(), S[:, -1].mean(), S[:, -1].std()])}
