import numpy as np
from scipy.optimize import minimize_scalar


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibration(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = abs(v-u-np.median(v-u, axis=0)) < 1.5
    axis = np.arange(u.shape[1])
    other = (axis+2) % len(axis)
    w = u[:, other]
    base = np.concatenate((np.broadcast_to(np.eye(len(axis)), (len(u), len(axis), len(axis))),
                           u[:, :, None]*np.eye(len(axis))), axis=2)

    def fit(freq):
        design = np.concatenate((base, np.stack((w, np.sin(freq*u), np.sin(freq*w)), axis=2)), axis=2)[keep]
        coef = _ls(design, v[keep])
        err = design @ coef - v[keep]
        return err @ err, coef

    freq = minimize_scalar(lambda f: fit(f)[0], bounds=(1.7, 2.65), method='bounded').x
    _, coeff = fit(freq)
    offset, scale = np.split(coeff[:-3], 2)
    coupling, own_sin, other_sin = coeff[-3:]
    z = (x-offset)/scale
    for _ in range(50):
        w = z[:, other]
        r = offset + scale*z + coupling*w + own_sin*np.sin(freq*z) + other_sin*np.sin(freq*w) - x
        if np.max(abs(r), initial=0) < 1e-11:
            break
        jac = np.zeros((len(x), len(axis), len(axis)))
        jac[:, axis, axis] = scale + own_sin*freq*np.cos(freq*z)
        jac[:, axis, other] = coupling + other_sin*freq*np.cos(freq*w)
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step / np.maximum(1., np.max(abs(step), axis=1, keepdims=True))
    return z


def _product(z):
    return z[:,[0,0,1,2]]*z[:,[1,3,2,3]]


def _cosine(z):
    return np.cos(2.4*z)


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    x = np.vstack((s['train']['x'], s['test']['x']))
    z = _calibration(s['panel_a'], x)
    b, c = s['panel_b'], s['panel_c']
    p, r = _product(z), _cosine(z)
    return dict(z=z, L=_ones(z) @ _ls(_ones(b['u']), b['v']),
                Q=_ones(p) @ _ls(_ones(_product(c['u'])), c['v1']),
                R=_ones(r) @ _ls(_ones(_cosine(c['u'])), c['v2']),
                product=p, cosine=r)
