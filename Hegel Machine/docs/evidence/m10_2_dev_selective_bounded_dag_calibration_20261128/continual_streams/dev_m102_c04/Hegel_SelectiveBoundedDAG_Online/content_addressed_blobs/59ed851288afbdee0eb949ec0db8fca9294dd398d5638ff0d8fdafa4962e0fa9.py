"""Invert the calibration panel and construct the spatial interaction features."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _products(x, pairs=_PAIRS):
    return np.column_stack([x[:, i] * x[:, j] for i, j in pairs])


def _coordinates(panel, x):
    u, v = panel['u'], panel['v']
    a = np.column_stack((_affine(u), u**3, _products(u)))
    keep = np.linalg.norm(v-u, axis=1) < 4.0
    if keep.sum() < a.shape[1]:
        keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        w = _fit(a[keep], v[keep])
        err = np.linalg.norm(a @ w - v, axis=1)
        new_keep = err <= max(1e-7, 6.0*np.median(err[keep]))
        if new_keep.sum() < a.shape[1] or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    active = np.linalg.norm(w[9:], axis=1) > max(1e-3, 0.01*np.linalg.norm(w[9:], axis=1).max())
    pairs = tuple(p for p, b in zip(_PAIRS, active) if b)
    cols = np.r_[np.ones(9, dtype=bool), active]
    w = _fit(a[keep][:, cols], v[keep])
    lin, cub, cross = w[1:5], w[5:9], w[9:]

    def value(q):
        z = w[0] + q @ lin + q**3 @ cub
        for (i, j), d in zip(pairs, cross):
            z += q[i]*q[j]*d
        return z

    def jac(q):
        jmat = lin + 3*q[:, None]**2*cub
        for (i, j), d in zip(pairs, cross):
            jmat[i] += q[j]*d
            jmat[j] += q[i]*d
        return jmat.T

    seeds = (x-w[0]) @ np.linalg.pinv(lin)
    return np.array([least_squares(lambda q: value(q)-obs, seed, jac=jac,
                                  ftol=1e-11, xtol=1e-11, gtol=1e-11).x
                     for obs, seed in zip(x, seeds)]).reshape(x.shape)


def _operator(u, v, query, second):
    def base(x, k):
        if second:
            return np.column_stack((_affine(x), _products(x), x**3/(1+k*x*x)))
        return np.column_stack((_affine(x), x/(1+k*np.abs(x))))

    def objective(k):
        a = base(u, k)
        return np.sum((a @ _fit(a, v)-v)**2)

    grid = np.linspace(0.02, 3.0, 61)
    idx = int(np.argmin([objective(k) for k in grid]))
    k = minimize_scalar(objective, bounds=(grid[max(0, idx-1)], grid[min(idx+1, len(grid)-1)]),
                        method='bounded', options={'xatol': 1e-9}).x
    if second:
        w = _fit(base(u, k), v)
        mags = np.linalg.norm(w[5:11], axis=1)
        active = mags > max(1e-3, 0.05*mags.max())
        cols = np.r_[np.ones(5, dtype=bool), active, np.ones(4, dtype=bool)]
        def reduced(k):
            a = base(u, k)[:, cols]
            return np.sum((a @ _fit(a, v)-v)**2)
        k = minimize_scalar(reduced, bounds=(max(0.001, k-0.2), k+0.2),
                            method='bounded', options={'xatol': 1e-9}).x
    else:
        cols = slice(None)
    return base(query, k)[:, cols] @ _fit(base(u, k)[:, cols], v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    q = _coordinates(study['panel_a'], x)
    panel = study['panel_c']
    p = _operator(panel['u'], panel['v1'], q, False)
    r = _operator(panel['u'], panel['v2'], q, True)
    joint = p[:, :, None]*r[:, None, :]
    common = np.column_stack((p, r, np.trace(joint, axis1=1, axis2=2)))
    offdiag = joint[:, ~np.eye(joint.shape[1], dtype=bool)]
    f = _affine(common)
    y_hat = f[n:] @ _fit(f[:n], train['y'])
    return {'common': common, 'offdiag': offdiag, 'y_hat': y_hat}
