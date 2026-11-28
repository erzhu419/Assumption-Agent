import numpy as np
from scipy.optimize import least_squares, root


def _linear(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def _ones(z):
    return np.column_stack((np.ones(len(z)), z))


def _pairs(z):
    return z * np.roll(z, -1, axis=1)


def _align(panel, x):
    u, v = panel['u'], panel['v']
    j = np.roll(np.arange(u.shape[1]), -1)
    dp = np.linalg.norm(v-u, axis=1)
    keep = dp <= np.median(dp) + max(0.5, 5*np.median(np.abs(dp-np.median(dp))))
    def design(z, a):
        return np.column_stack((np.ones(len(z)), z[:, a], z[:, j[a]], z[:, a]*z[:, j[a]], z[:, a]**3))
    for _ in range(3):
        coef = np.array([_linear(design(u[keep], a), v[keep, a]) for a in range(u.shape[1])])
        residual = np.linalg.norm(np.column_stack([design(u, a)@coef[a] for a in range(u.shape[1])])-v, axis=1)
        sel = residual <= max(0.003, 6*np.median(residual))
        if np.array_equal(sel, keep) or sel.sum() < 8:
            break
        keep = sel
    a, b, c, d, e = coef.T
    def obs(z):
        return a + b*z + c*z[j] + d*z*z[j] + e*z**3
    return np.asarray([root(lambda z: obs(z)-row, (row-a)/b).x for row in x])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    n = len(study['train']['x'])
    x = np.vstack((study['train']['x'], study['test']['x']))
    z = _align(study['panel_a'], x)
    panel = study['panel_c']
    u = panel['u']
    c = _ones(_pairs(z)) @ _linear(_ones(_pairs(u)), panel['v1'])
    def cosine(x, q):
        return _ones(np.cos(q*x))
    def residual(q):
        f = cosine(u, q)
        return (f@_linear(f, panel['v2'])-panel['v2']).ravel()
    q = least_squares(residual, [2.2], bounds=(0.5, 4.0)).x
    d = cosine(z, q) @ _linear(cosine(u, q), panel['v2'])
    w = np.column_stack((c, d))
    return {'features': (w-w[:n].mean(0))/np.maximum(w[:n].std(0), 0.1), 'calibrated': w}
