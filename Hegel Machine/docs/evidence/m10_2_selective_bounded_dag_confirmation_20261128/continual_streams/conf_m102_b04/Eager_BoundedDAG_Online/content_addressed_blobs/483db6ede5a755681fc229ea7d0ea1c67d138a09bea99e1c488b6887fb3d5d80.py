"""Recover latent coordinates from the reference panel, then fit each outcome."""
from itertools import combinations_with_replacement
import numpy as np


def _calibrate(panel, x, seed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    pair = np.arange(u.shape[1]) ^ 2
    # Gross corruption is rowwise. Remove only the separated residual tail.
    a = np.column_stack((np.ones(len(u)), u))
    keep = np.arange(len(u))
    for _ in range(4):
        coef = np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]
        residual = np.linalg.norm(v - a @ coef, axis=1)
        cutoff = 4 * np.median(residual)
        keep = np.flatnonzero(residual <= max(cutoff, 1e-6))
    u, v = u[keep], v[keep]

    def fit(freq):
        w, h = freq
        cols = []
        err = 0.
        for j, k in enumerate(pair):
            f = np.column_stack((np.ones(len(u)), u[:, j], np.sin(w*u[:, k]), np.sin(h*u[:, j])))
            c = np.linalg.lstsq(f, v[:, j], rcond=None)[0]
            cols.append(c)
            err += np.sum((f @ c - v[:, j])**2)
        return err, np.asarray(cols).T

    # Profile out the linear amplitudes and refine the two shared frequencies.
    freq = np.array([2.4, 3.1])
    for radius in (.45, .15, .045, .015, .005, .0015):
        for j in range(2):
            trial = [freq + shift * np.eye(2)[j] for shift in np.linspace(-radius, radius, 9)]
            freq = min(trial, key=lambda p: fit(p)[0])
    w, h = freq
    _, (bias, scale, cross, own) = fit(freq)

    # Invert the coupled map, initialized from the frozen affine coordinates.
    q = np.array(seed, dtype=float, copy=True)
    for _ in range(25):
        f = bias + scale*q + cross*np.sin(w*q[:, pair]) + own*np.sin(h*q) - x
        if np.max(np.abs(f), initial=0.) < 1e-11:
            break
        jac = np.zeros((len(q), q.shape[1], q.shape[1]))
        j = np.arange(q.shape[1])
        jac[:, j, j] = scale + own*h*np.cos(h*q)
        jac[:, j, pair] = cross*w*np.cos(w*q[:, pair])
        step = np.linalg.solve(jac, f[..., None])[..., 0]
        q -= np.clip(step, -.5, .5)
    return q


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    features = inputs['node_00']
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    d = x.shape[1]
    k = features['aligned'].shape[1]
    seed = features['hybrid'][:, k+1:k+1+d]
    q = _calibrate(study['panel_a'], x, seed)
    a = np.column_stack((np.ones(len(q)), q))
    b = np.column_stack([a] + [q[:, i]*q[:, j] for i, j in combinations_with_replacement(range(d), 2)])
    y = b[n:] @ np.linalg.lstsq(b[:n], train['y'], rcond=None)[0]
    state = a[n:] @ np.linalg.lstsq(a[:n], train['traj'], rcond=None)[0]
    diag = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diag}
