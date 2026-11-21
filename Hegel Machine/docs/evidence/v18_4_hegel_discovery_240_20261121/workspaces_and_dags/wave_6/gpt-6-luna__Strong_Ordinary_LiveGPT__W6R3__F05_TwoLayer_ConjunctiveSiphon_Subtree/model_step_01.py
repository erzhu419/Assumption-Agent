import math
import numpy as np
from scipy.optimize import least_squares

def _base(p, row, branch):
    vmax, rho, c = p
    un = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    core = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    if branch == 'sqrt':
        return core + c * math.sqrt(un)
    return core - c * un

def fit(train_rows):
    best = None
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.array([_base(p, r, branch) - float(r['y']) for r in train_rows])
        result = least_squares(residual, x0=np.array([4.0, 0.7, 0.1]), bounds=([0.0, 0.0, -10.0], [100.0, 2.0, 10.0]))
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, branch, result.x.copy())
    return {'branch': best[1], 'params': best[2]}

def predict(row):
    return float(_base(_MODEL['params'], row, _MODEL['branch']))