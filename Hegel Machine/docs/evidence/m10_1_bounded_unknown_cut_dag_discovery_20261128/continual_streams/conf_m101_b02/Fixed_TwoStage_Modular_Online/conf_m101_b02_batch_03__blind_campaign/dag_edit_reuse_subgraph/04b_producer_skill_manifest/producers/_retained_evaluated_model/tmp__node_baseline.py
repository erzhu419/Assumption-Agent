from itertools import combinations

import numpy as np


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    j = np.roll(u, -1, axis=1)
    designs = [np.column_stack((np.ones(len(u)), p, q, p*q, p**3))
               for p, q in zip(u.T, j.T)]
    coeff = np.empty((5, u.shape[1]))
    weight = np.ones(len(u))
    for _ in range(24):
        residual = np.empty_like(v)
        for i, a in enumerate(designs):
            coeff[:, i] = np.linalg.lstsq(a * weight[:, None], v[:, i] * weight,
                                         rcond=None)[0]
            residual[:, i] = v[:, i] - a @ coeff[:, i]
        radius = np.linalg.norm(residual, axis=1)
        scale = max(np.median(radius), 1e-8)
        weight = 1 / np.hypot(1, radius / scale)

    b, d, e, f, c = coeff
    idx = np.arange(u.shape[1])
    initial = np.diag(d)
    initial[idx, (idx + 1) % len(idx)] = e
    z = np.linalg.solve(initial, (x - b).T).T
    for _ in range(16):
        q = np.roll(z, -1, axis=1)
        r = b + d*z + e*q + f*z*q + c*z**3 - x
        jac = np.zeros((len(x), len(idx), len(idx)))
        jac[:, idx, idx] = d + f*q + 3*c*z*z
        jac[:, idx, (idx + 1) % len(idx)] = e + f*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step / np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return z


def _predict(u, y, z, feature):
    return feature(z) @ np.linalg.lstsq(feature(u), y, rcond=None)[0]


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _saturating(u):
    return np.column_stack((_affine(u), np.arctan(2.4 * u)))


def _square(u):
    return np.column_stack((_affine(u), u*u))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    b, c = study['panel_b'], study['panel_c']
    fd = _predict(b['u'], b['v'], z, _affine)
    cd = np.column_stack((_predict(c['u'], c['v1'], z, _saturating),
                          _predict(c['u'], c['v2'], z, _square)))
    cross = [z[:, i] * z[:, j] for i, j in combinations(range(z.shape[1]), 2)]
    g = np.column_stack((_square(z), x, *cross, np.arctan(2.4*z)))
    return {'g': g, 'fd': fd, 'cd': cd, 'y': train['y'], 'traj': train['traj']}
