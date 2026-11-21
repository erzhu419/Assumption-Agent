import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    branches = ['sqrt_plus', 'sqrt_minus', 'linear_leak']
    best = None
    for branch in branches:
        def pred(p, r):
            vmax, rho, c = p
            un = r['u_now'] / 4.0
            base = vmax * (1.0 - rho * r['s_grad'] / (1.0 + r['s_grad'])) * (un / (1.2 + un))
            if branch == 'sqrt_plus':
                return base + c * math.sqrt(max(un, 0.0))
            if branch == 'sqrt_minus':
                return base - c * math.sqrt(max(un, 0.0))
            return base - c * un
        def residual(p):
            return np.array([pred(p, r) - r['y'] for r in train_rows])
        result = least_squares(residual, [4.0, 0.5, 0.05], bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]))
        err = float(np.mean(residual(result.x) ** 2))
        if best is None or err < best[0]:
            best = (err, branch, result.x)
    _, branch, params = best
    def predict(row):
        vmax, rho, c = params
        un = row['u_now'] / 4.0
        base = vmax * (1.0 - rho * row['s_grad'] / (1.0 + row['s_grad'])) * (un / (1.2 + un))
        if branch == 'sqrt_plus':
            return float(base + c * math.sqrt(max(un, 0.0)))
        if branch == 'sqrt_minus':
            return float(base - c * math.sqrt(max(un, 0.0)))
        return float(base - c * un)
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) and use its returned predictor')