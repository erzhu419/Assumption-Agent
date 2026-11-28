import numpy as np
from itertools import combinations_with_replacement


def _basis(x, pairs):
    i, j = pairs
    return np.column_stack((np.ones(len(x)), x, x[:, i]*x[:, j], x**3))


def _calibrate(u, v):
    d = u.shape[1]
    pairs = np.array(list(combinations_with_replacement(range(d), 2))).T
    a = _basis(u, pairs)
    # Cauchy weights identify the three contaminated rows jointly across channels.
    w = np.ones(len(u))
    for _ in range(20):
        c = np.linalg.lstsq(a*w[:, None], v*w[:, None], rcond=None)[0]
        r = np.linalg.norm(a@c-v, axis=1)
        scale = max(2*np.median(r), 1e-9)
        w = scale/np.hypot(scale, r)
    keep = np.argsort(r)[:-3]
    return pairs, np.linalg.lstsq(a[keep], v[keep], rcond=None)[0]


def _invert(y, pairs, c):
    d = y.shape[1]
    linear = c[1:1+d].T
    x = np.linalg.solve(linear, (y-c[0]).T).T
    for _ in range(30):
        r = _basis(x, pairs)@c-y
        if np.max(np.abs(r)) < 1e-11:
            break
        jac = np.broadcast_to(linear, (len(x), d, d)).copy()
        for (i, j), coef in zip(pairs.T, c[1+d:-d]):
            jac[:, :, i] += x[:, j, None]*coef
            jac[:, :, j] += x[:, i, None]*coef
        jac += c[-d:].T[None, :, :]*(3*x[:, None, :]**2)
        step = np.linalg.solve(jac, r[:, :, None])[:, :, 0]
        trial = x-step
        norm = np.linalg.norm(r, axis=1)
        for _ in range(12):
            bad = np.linalg.norm(_basis(trial, pairs)@c-y, axis=1) > norm
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial[bad] = x[bad]-step[bad]
        x = trial
    return x


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, c = study['panel_a'], study['panel_c']
    pairs, coef = _calibrate(a['u'], a['v'])
    measured = np.concatenate((study['train']['x'], study['test']['x']))
    return {'u': c['u'], 'v1': c['v1'], 'v2': c['v2'], 'x': _invert(measured, pairs, coef)}
