import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_BRANCH = None

def fit(train_rows):
    global _PARAMS, _BRANCH
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    un = np.array([max(1e-8, float(r['u_now'])) / 4.0 for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    base = lambda vmax, rho: vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    candidates = []
    # Open local branch: additive square-root source term.
    def res_sqrt(p):
        return base(p[0], p[1]) + p[2] * np.sqrt(un) - y
    r1 = least_squares(res_sqrt, [5.0, 0.5, 0.1], bounds=([0.0, 0.0, -10.0], [100.0, 5.0, 10.0]), max_nfev=30000)
    candidates.append((float(np.sum(res_sqrt(r1.x)**2)), r1.x, 'sqrt'))
    # Alternative mass-drift branch: linear leak term.
    def res_leak(p):
        return base(p[0], p[1]) - p[2] * un - y
    r2 = least_squares(res_leak, [8.0, 0.5, 0.1], bounds=([0.0, 0.0, -10.0], [100.0, 5.0, 10.0]), max_nfev=30000)
    candidates.append((float(np.sum(res_leak(r2.x)**2)), r2.x, 'leak'))
    _, _PARAMS, _BRANCH = min(candidates, key=lambda item: item[0])
    return {'branch': _BRANCH, 'parameters': _PARAMS.tolist()}

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    vmax, rho, branch_coeff = [float(v) for v in _PARAMS]
    un = max(1e-8, float(row['u_now'])) / 4.0
    s = max(0.0, float(row['s_grad']))
    value = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    if _BRANCH == 'sqrt':
        value += branch_coeff * math.sqrt(un)
    else:
        value -= branch_coeff * un
    return float(value)
