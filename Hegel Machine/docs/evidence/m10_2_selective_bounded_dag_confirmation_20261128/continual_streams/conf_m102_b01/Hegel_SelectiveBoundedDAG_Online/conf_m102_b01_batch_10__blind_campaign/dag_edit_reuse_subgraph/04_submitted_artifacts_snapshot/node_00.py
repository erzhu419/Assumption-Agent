import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _panel_a_design(x, h):
    return _aff(np.column_stack((x, np.sin(h[0]*x), np.tanh(h[1]*x))))


def _panel_b_design(x, h):
    phase = h[0]*x
    return _aff(x + h[1]*np.sin(phase) + h[2]*np.cbrt(x) + h[3]*(1-np.cos(phase)))


def _calibrate(design, u, v, starts, budget):
    def residual(h):
        a = design(u, h)
        return (a @ _ls(a, v) - v).ravel()

    fits = (least_squares(residual, start, max_nfev=budget) for start in starts)
    h = min(fits, key=lambda fit: fit.cost).x
    return h, _ls(design(u, h), v)


def _invert_a(x, h, coef):
    linear, sine, sat = coef[1:5], coef[5:9], coef[9:13]
    z = (x-coef[0]) @ np.linalg.pinv(linear)
    for _ in range(35):
        t = np.tanh(h[1]*z)
        phase = h[0]*z
        err = coef[0] + z@linear + np.sin(phase)@sine + t@sat - x
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        jac = linear + (h[0]*np.cos(phase))[:, :, None]*sine + (h[1]*(1-t*t))[:, :, None]*sat
        step = np.linalg.solve(jac.transpose(0, 2, 1)+1e-8*np.eye(4), err[..., None])[..., 0]
        z -= np.clip(step, -.2, .2)
    return z


def run_node(inputs: dict) -> dict:
    source = inputs['__input__']
    study = source['study'] if 'study' in source else source
    a = study['panel_a']
    u, v = np.asarray(a['u']), np.asarray(a['v'])
    f = _aff(u)
    err = np.linalg.norm(f@_ls(f, v)-v, axis=1)
    good = err <= max(0.8, 6*np.median(err))
    if np.count_nonzero(good) < 14:
        good = np.ones(len(u), dtype=bool)
    ha, ca = _calibrate(_panel_a_design, u[good], v[good], [(2.4, 1.8)], 150)
    b = study['panel_b']
    starts = [(s, a, 0., 0.) for s in (1.25, 2.15, 2.9, 4., 5.5) for a in (-.4, .4)]
    hb, cb = _calibrate(_panel_b_design, np.asarray(b['u']), np.asarray(b['v']), starts, 90)
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    return {'w': _panel_b_design(_invert_a(x, ha, ca), hb) @ cb}
