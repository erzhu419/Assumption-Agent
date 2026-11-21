import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def _base(row, p, branch):
    vmax, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # tau_scale = 1.0
    core = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    if branch == 'sqrt_plus':
        return core + c * math.sqrt(max(un, 0.0))
    if branch == 'sqrt_minus':
        return core - c * math.sqrt(max(un, 0.0))
    return core - c * un

def fit(train_rows):
    global _PARAMS, _BRANCH
    y = np.asarray([float(r['y']) for r in train_rows])
    choices = [('sqrt_plus', [2.0, 0.5, 1.0]), ('sqrt_minus', [2.0, 0.5, 0.2]), ('linear_leak', [2.0, 0.5, 0.2])]
    best = None
    for branch, init in choices:
        def residual(p):
            return np.asarray([_base(r, p, branch) for r in train_rows]) - y
        sol = least_squares(residual, init, bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]), max_nfev=20000)
        err = float(np.dot(sol.fun, sol.fun))
        if best is None or err < best[0]:
            best = (err, branch, sol.x)
    _, _BRANCH, _PARAMS = best
    return {'branch': _BRANCH, 'parameters': _PARAMS.tolist()}

def predict(row) -> float:
    return float(_base(row, _PARAMS, _BRANCH))