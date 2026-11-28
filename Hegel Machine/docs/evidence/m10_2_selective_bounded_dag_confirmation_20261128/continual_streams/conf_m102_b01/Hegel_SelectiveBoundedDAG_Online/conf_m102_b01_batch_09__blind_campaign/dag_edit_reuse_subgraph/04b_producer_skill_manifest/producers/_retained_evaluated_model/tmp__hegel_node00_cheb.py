import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebval, chebder


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    # Gross contamination affects whole calibration rows.
    center = np.median(v, axis=0)
    mad = np.maximum(np.median(abs(v - center), axis=0), 1e-8)
    keep = np.max(abs(v - center) / mad, axis=1) < 6
    u, v = u[keep], v[keep]
    n, p = u.shape
    eye = np.tile(np.eye(p), (n, 1))
    t = u.ravel()
    z = np.roll(u, -2, axis=1).ravel()
    radius = 1.3
    d = np.column_stack((eye, eye * t[:, None], chebvander(t / radius, 9)[:, 3::2], chebvander(z / radius, 9)[:, 1::2]))
    q = np.linalg.lstsq(d, v.ravel(), rcond=None)[0]
    a, b = np.zeros(10), np.zeros(10)
    a[3::2], b[1::2] = q[2*p:2*p+4], q[2*p+4:]
    return q[:p], q[p:2*p], a, b, radius


def _inverse(x, fit):
    off, slope, a, b, radius = fit
    da, db = chebder(a) / radius, chebder(b) / radius
    def evaluate(z, c, dc):
        t = np.clip(z, -radius, radius)
        grad = chebval(t / radius, dc)
        return chebval(t / radius, c) + grad * (z - t), grad
    z = (x - off) / slope
    j = np.arange(x.shape[1])
    jac = np.zeros((len(x), len(j), len(j)))
    for _ in range(24):
        self_term, self_grad = evaluate(z, a, da)
        cross_term, cross_grad = evaluate(np.roll(z, -2, axis=1), b, db)
        residual = off + slope * z + self_term + cross_term - x
        if np.max(abs(residual), initial=0.) < 1e-12:
            break
        jac[:, j, j] = slope + self_grad
        jac[:, j, (j + 2) % len(j)] = cross_grad
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        z -= np.clip(step, -0.5, 0.5)
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    z = _inverse(x, _calibrate(study['panel_a']))
    b = study['panel_b']
    return {'w': _aff(z) @ np.linalg.lstsq(_aff(b['u']), b['v'], rcond=None)[0]}
