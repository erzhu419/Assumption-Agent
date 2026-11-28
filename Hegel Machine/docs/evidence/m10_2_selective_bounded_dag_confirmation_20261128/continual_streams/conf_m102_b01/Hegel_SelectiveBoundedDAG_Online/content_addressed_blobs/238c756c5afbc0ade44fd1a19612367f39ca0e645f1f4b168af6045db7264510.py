import numpy as np


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    n, p = u.shape
    eye = np.tile(np.eye(p), (n, 1))
    z = u.ravel()
    opposite = np.roll(u, -2, axis=1).ravel()
    d = np.column_stack((eye, eye * z[:, None], np.tanh(1.8 * z), np.sin(2.4 * opposite)))
    target = v.ravel()

    def fit(rows):
        k = np.repeat(rows, p)
        return np.linalg.lstsq(d[k], target[k], rcond=None)[0]

    center = np.median(v, axis=0)
    mad = np.maximum(np.median(np.abs(v - center), axis=0), 1e-8)
    good = np.max(np.abs(v - center) / mad, axis=1) < 6
    if good.sum() < n // 2:
        good[:] = True
    q = fit(good)
    # Refine on whole rows using forward residuals, not response magnitude.
    for _ in range(8):
        err = np.linalg.norm((d @ q - target).reshape(n, p), axis=1)
        k = max(p + 2, int(np.ceil(0.8 * n)))
        keep = np.argsort(err)[:k]
        good = np.zeros(n, dtype=bool)
        good[keep] = True
        q = fit(good)
    err = np.linalg.norm((d @ q - target).reshape(n, p), axis=1)
    q = fit(err <= max(8 * np.median(err), 1e-3))
    return q[:p], q[p:2*p], q[-2], q[-1]


def _forward(z, fit):
    off, slope, a, b = fit
    return off + slope * z + a * np.tanh(1.8 * z) + b * np.sin(2.4 * np.roll(z, -2, axis=1))


def _inverse(x, fit):
    off, slope, a, b = fit
    z = (x - off) / slope
    j = np.arange(x.shape[1])
    jac = np.zeros((len(x), len(j), len(j)))
    for _ in range(32):
        h = np.tanh(1.8 * z)
        cross = 2.4 * np.roll(z, -2, axis=1)
        residual = off + slope * z + a * h + b * np.sin(cross) - x
        if np.max(np.abs(residual), initial=0.) < 1e-12:
            break
        jac[:, j, j] = slope + 1.8 * a * (1 - h * h)
        jac[:, j, (j + 2) % len(j)] = 2.4 * b * np.cos(cross)
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= np.clip(step, -0.5, 0.5)
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    z = _inverse(x, _calibrate(study['panel_a']))
    b = study['panel_b']
    return {'w': _aff(z) @ np.linalg.lstsq(_aff(b['u']), b['v'], rcond=None)[0]}
