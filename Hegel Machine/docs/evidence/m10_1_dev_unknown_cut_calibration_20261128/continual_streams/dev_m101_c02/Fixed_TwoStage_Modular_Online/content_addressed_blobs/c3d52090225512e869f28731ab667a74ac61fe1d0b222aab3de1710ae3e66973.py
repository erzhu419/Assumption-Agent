import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares, minimize_scalar

_TERMS = tuple(t for d in (1, 2, 3) for t in combinations_with_replacement(range(4), d))

def _poly(x):
    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, t], axis=1) for t in _TERMS])

def _fit_a(u, v):
    X = _poly(u)
    Q = X[:, :15]
    w = np.ones(len(u))
    for _ in range(30):
        beta = np.linalg.lstsq(Q*w[:, None], v*w[:, None], rcond=None)[0]
        r = np.linalg.norm(Q@beta-v, axis=1)
        w = 1/np.hypot(1, r/max(0.001, np.median(r)))
    keep = np.argsort(r)[:-3]
    return np.linalg.lstsq(X[keep], v[keep], rcond=None)[0]


def _invert_a(v, b):
    x = np.linalg.solve(b[1:5].T, (v-b[0]).T).T
    for _ in range(30):
        r = _poly(x)@b-v
        jac = np.zeros((len(v), 4, 4))
        for t, c in zip(_TERMS, b[1:]):
            for j, k in enumerate(t):
                m = t[:j]+t[j+1:]
                jac[:, :, k] += np.prod(x[:, m], axis=1)[:, None]*c
        dx = np.linalg.solve(jac, r[..., None])[..., 0]
        if np.max(np.abs(dx)) < 1e-11:
            break
        target = np.sum(r*r, axis=1)
        for _ in range(12):
            bad = np.sum((_poly(x-dx)@b-v)**2, axis=1) > target
            if not np.any(bad):
                break
            dx[bad] *= 0.5
        x -= dx
    return x

def _basis_b(x, omega):
    return np.column_stack((np.ones(len(x)), x, np.sin(omega*x)))

def _warp_design(x, omega, gain):
    return np.column_stack((np.ones(len(x)), x + gain*np.sin(omega*x)))


def _fit_b(u, v):
    # Initialize the shared warp from the frequency of the unrestricted harmonic fit.
    def profile(omega):
        X = _basis_b(u, omega)
        b = np.linalg.lstsq(X, v, rcond=None)[0]
        return np.sum((X@b-v)**2), b
    grid = np.linspace(0.5, 5, 46)
    w = min(grid, key=lambda s: profile(s)[0])
    w = minimize_scalar(lambda s: profile(s)[0], bounds=(w-0.1, w+0.1), method='bounded').x

    def shared_profile(params):
        X = _warp_design(u, *params)
        b = np.linalg.lstsq(X, v, rcond=None)[0]
        return (X@b-v).ravel(), b

    params = least_squares(lambda t: shared_profile(t)[0], [w, -0.4]).x
    return params, shared_profile(params)[1]

def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    coeff_a = _fit_a(a['u'], a['v'])
    warp, coeff_b = _fit_b(b['u'], b['v'])
    def calibrate(x):
        return _warp_design(_invert_a(x, coeff_a), *warp)@coeff_b
    return {'z_tr': calibrate(study['train']['x']), 'z_te': calibrate(study['test']['x'])}
