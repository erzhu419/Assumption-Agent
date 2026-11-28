"""Robust trajectory and readout calibration in the frozen spatial basis."""
import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares


def _regress(design, target, penalty, robust=False):
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-10)
    a = design / scale
    p = np.diag(np.broadcast_to(penalty, (a.shape[1],)))
    out = solve(a.T @ a + p, a.T @ target, assume_a='pos')
    if robust:
        for _ in range(12):
            err = np.sqrt(np.mean((a @ out - target)**2, axis=1))
            tau = max(2 * np.median(err), .006)
            w = 1 / (1 + (err / tau)**2)
            out = solve(a.T @ (a * w[:, None]) + p, a.T @ (target * w[:, None]), assume_a='pos')
    return out / (scale if target.ndim == 1 else scale[:, None])


def _state_map(study, x):
    panel = study['panel_b']
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    def design(f):
        return np.c_[np.ones(len(u)), u, np.sin(f[0]*u)]
    def residual(f):
        a = design(f)
        return (a @ np.linalg.lstsq(a, v, rcond=1e-10)[0] - v).ravel()
    result = min((least_squares(residual, [a], bounds=(.3,9)) for a in (1.,2.9,5.5,8.)), key=lambda o:o.cost)
    z = x[:, :4]
    w = z - .4*np.sin(result.x[0]*z)
    j = np.eye(x.shape[1]); q, r = np.triu_indices(4)
    ab = np.r_[j[4:5], j[25:29]].T
    c = np.linalg.lstsq(x @ ab, w, rcond=1e-10)[0]
    linear = ab @ c
    squared = (j[4:5].T * c[:1]**2 + j[25:29].T @ (2*c[:1]*c[1:]) +
               j[15:25].T @ ((1+(q!=r))[:,None]*c[1:][q]*c[1:][r]))
    return np.c_[j[:5].T, j[[5,9,12,14]].T, squared, linear]


def run_node(inputs: dict) -> dict:
    study, features = inputs['__input__'], inputs['node_00']
    train = study['train']
    r = np.asarray(features['readout_train'], float)
    x = np.asarray(features['z'], float)
    readout = _regress(r, np.asarray(train['y'],float).reshape(-1),
                      np.r_[np.full(4,.0001),np.full(10,.3),1e-7,np.full(10,.3),np.full(16,.001),np.full(4,.0001)])
    transform = _state_map(study, x)
    coef = transform @ _regress(x @ transform, np.asarray(train['traj'],float), .04642, True)
    return {'readout_coef': readout, 'state_coef': coef}
