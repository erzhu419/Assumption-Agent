"""Panel-calibrated predictor for the campaign's scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    out = np.empty_like(x, dtype=float)
    for j, (u, v) in enumerate(zip(panel['u'].T, panel['v'].T)):
        d = _aff(u)
        b = _ls(d, v)
        for _ in range(8):
            keep = np.argsort(np.abs(d @ b - v))[:int(.875 * len(u))]
            b = _ls(d[keep], v[keep])
        e = np.abs(d @ b - v)
        keep = e <= max(10 * np.median(e), .002)
        b = _ls(d[keep], v[keep])
        out[:, j] = (x[:, j] - b[0]) / b[1]
    return out


def _panel(u, v, feature, init, bounds):
    def error(p):
        a = _aff(feature(u, p))
        return (a @ _ls(a, v) - v).ravel()
    p = least_squares(error, init, bounds=bounds, max_nfev=300).x
    b = _ls(_aff(feature(u, p)), v)
    return lambda x: _aff(feature(x, p)) @ b


def _b(x, p):
    threshold, exponent = p
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.)**exponent


def _c(x, p):
    a, b, c, d = p
    return x/(1 + a*np.abs(x) + b*x*x + c*np.abs(x)**3) + d*x


def _latent(study, x):
    b, c = study['panel_b'], study['panel_c']
    f = _panel(b['u'], b['v'], _b, [.35, 1.5], ([0., .25], [.8, 4.]))
    g = _panel(c['u'], c['v1'], _c, [.98,.37,.05,.087], ([0,0,0,-1], [20,20,20,1]))
    return g(f(_calibrate(study['panel_a'], x)))


def _scalar_design(z):
    return np.c_[_aff(z), z[:,0]*z[:,2], z[:,1]*z[:,3]]


def solve_batch(study: dict) -> dict:
    tr = study['train']
    z = _latent(study, np.vstack((tr['x'], study['test']['x'])))
    n = len(tr['x'])
    d = _scalar_design(z)
    y = d[n:] @ _ls(d[:n], tr['y'])
    d = _aff(z)
    # Use only training trajectories to estimate the two shared temporal modes.
    modes = np.linalg.svd(tr['traj'], full_matrices=False)[2][:2]
    pred = (d[n:] @ _ls(d[:n], tr['traj'] @ modes.T)) @ modes
    return {'y_hat': y, 'state_pred': pred,
            'diagnostics_vec': np.array([y.mean(), y.std(), pred[:,-1].mean(), pred[:,-1].std()])}
