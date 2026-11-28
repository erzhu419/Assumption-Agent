"""Fit a nonlinear scalar readout and a separate affine trajectory."""
import numpy as np
from scipy.optimize import least_squares


def _affine(c):
    return np.column_stack((np.ones(len(c)), c))


def _readout(c, query, y):
    x, xq = _affine(c), _affine(query)
    j, k = np.triu_indices(c.shape[1])
    a = np.column_stack((x, c[:, j]*c[:, k]))
    aq = np.column_stack((xq, query[:, j]*query[:, k]))
    q = np.linalg.svd(a, full_matrices=False)[0]
    r = y - q @ (q.T @ y)
    rng = np.random.default_rng(6)
    starts = [rng.normal(0, [.35,.8,1.6][i%3], x.shape[1]) for i in range(16)]
    best = None
    for f in (np.sin, np.tanh, lambda t: t/(1+t*t)):
        def residual(p):
            h = f(x @ p)
            h -= q @ (q.T @ h)
            return h * (h @ r / max(h @ h, 1e-20)) - r
        for p in starts:
            opt = least_squares(residual, np.clip(p,-3.9,3.9), bounds=(-4.,4.), max_nfev=130)
            if best is None or opt.cost < best[0]:
                best = opt.cost, f, opt.x
    _, f, p = best
    design = np.column_stack((a, f(x @ p)))
    beta = np.linalg.lstsq(design, y, rcond=1e-10)[0]
    return np.column_stack((aq, f(xq @ p))) @ beta


def run_node(inputs: dict) -> dict:
    study, upstream = inputs['__input__'], inputs['node_00']
    c, cq = upstream['train_coords'], upstream['query_coords']
    train = study['train']
    beta = np.linalg.lstsq(_affine(c), np.asarray(train['traj'], float), rcond=None)[0]
    return {'beta': beta, 'new_design': _affine(cq), 't': np.asarray(study['eval_times'], float),
            'y_hat': _readout(c, cq, np.asarray(train['y'], float).reshape(-1))}
