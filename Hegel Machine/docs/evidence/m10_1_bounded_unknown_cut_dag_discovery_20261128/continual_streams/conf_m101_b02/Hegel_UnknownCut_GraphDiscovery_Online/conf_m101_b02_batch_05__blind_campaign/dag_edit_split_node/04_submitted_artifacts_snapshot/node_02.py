import numpy as np


def _design(u):
    return np.stack((np.ones_like(u), u, np.roll(u, -1, axis=1),
                     u * np.roll(u, -1, axis=1), u**3), axis=-1)


def _calibrate(u, v):
    basis = _design(u)
    weight = np.ones(len(u))
    for _ in range(18):
        coeff = np.stack([np.linalg.lstsq(basis[:, j] * weight[:, None],
                                         v[:, j] * weight, rcond=None)[0]
                          for j in range(u.shape[1])])
        residual = np.linalg.norm(np.sum(basis * coeff, axis=-1) - v, axis=1)
        scale = max(np.median(residual), 1e-8)
        weight = 1 / np.hypot(1, residual / scale)
    return coeff


def _invert(x, coeff):
    j = np.arange(x.shape[1])
    k = (j + 1) % len(j)
    linear = np.diag(coeff[:, 1])
    linear[j, k] = coeff[:, 2]
    z = np.linalg.solve(linear, (x - coeff[:, 0]).T).T
    jac = np.zeros((len(x), len(j), len(j)))
    for _ in range(24):
        r = np.sum(_design(z) * coeff, axis=-1) - x
        if np.max(np.abs(r)) < 1e-11:
            break
        jac[:, j, j] = coeff[:, 1] + coeff[:, 3] * z[:, k] + 3 * coeff[:, 4] * z**2
        jac[:, j, k] = coeff[:, 2] + coeff[:, 3] * z
        dz = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= dz / np.maximum(1, np.max(np.abs(dz), axis=1, keepdims=True))
    return z


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    a, b = s['panel_a'], s['panel_b']
    coeff = _calibrate(a['u'], a['v'])
    x = np.vstack((s['train']['x'], s['test']['x']))
    return {'z': _invert(x, coeff), 'b_u': b['u'], 'b_v': b['v']}
