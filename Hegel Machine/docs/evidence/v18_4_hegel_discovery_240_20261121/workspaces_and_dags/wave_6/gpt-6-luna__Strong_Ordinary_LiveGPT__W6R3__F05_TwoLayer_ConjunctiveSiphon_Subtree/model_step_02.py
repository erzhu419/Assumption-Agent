import math
import numpy as np
from scipy.optimize import least_squares

_MODEL = None

def _base(p, row, branch):
    vmax, rho, c = p
    un = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    core = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    if branch == 'sqrt_plus':
        return core + c * math.sqrt(max(un, 0.0))
    if branch == 'sqrt_minus':
        return core - c * math.sqrt(max(un, 0.0))
    return core - c * un

def fit(train_rows):
    global _MODEL
    candidates = []
    for branch in ('sqrt_plus', 'sqrt_minus', 'leak'):
        def residual(p):
            return np.array([_base(p, r, branch) - float(r['y']) for r in train_rows])
        result = least_squares(residual, x0=np.array([4.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 20.0]), max_nfev=20000)
        candidates.append((float(np.sum(result.fun ** 2)), branch, result.x))
    _, branch, params = min(candidates, key=lambda item: item[0])
    _MODEL = {'branch': branch, 'params': params.tolist()}
    return _MODEL

def predict(row) -> float:
    if _MODEL is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_base(_MODEL['params'], row, _MODEL['branch']))