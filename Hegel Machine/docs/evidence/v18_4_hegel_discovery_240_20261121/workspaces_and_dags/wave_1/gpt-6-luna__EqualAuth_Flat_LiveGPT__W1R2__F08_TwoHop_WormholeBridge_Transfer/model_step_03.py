import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p, branch):
    vmax, rho, c = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c * (math.sqrt(u) if branch == 'sqrt' else -u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.array([_value(r, p, branch) for r in train_rows]) - y
        result = least_squares(residual, x0=[4.0, 0.7, 0.2], bounds=([0.0, 0.0, 0.0], [100.0, 1.0, 100.0]))
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result.x, branch)
    return {'params': best[1].tolist(), 'branch': best[2]}

def predict(row):
    return float(_value(row, _MODEL['params'], _MODEL['branch']))
