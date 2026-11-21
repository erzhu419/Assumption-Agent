import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    tau = 4.0
    def predict_with(p, row, branch):
        vmax, rho, c = p
        u = max(1e-8, float(row['u_now']) / tau)
        s = max(0.0, float(row['s_grad']))
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        if branch == 'sqrt_plus':
            return base + c * math.sqrt(u)
        if branch == 'sqrt_minus':
            return base - c * math.sqrt(u)
        return base - c * u
    candidates = []
    for branch in ('sqrt_plus', 'sqrt_minus', 'linear_leak'):
        def residual(p):
            return np.array([predict_with(p, r, branch) - float(r['y']) for r in rows])
        result = least_squares(residual, x0=[5.0, 0.7, 0.2], bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]), max_nfev=5000)
        candidates.append((float(np.mean(residual(result.x) ** 2)), branch, result.x))
    _, branch, params = min(candidates, key=lambda item: item[0])
    return {'branch': branch, 'params': np.asarray(params, dtype=float), 'tau': tau}

def predict(row):
    model = fit(train_rows)
    vmax, rho, c = model['params']
    u = max(1e-8, float(row['u_now']) / model['tau'])
    s = max(0.0, float(row['s_grad']))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if model['branch'] == 'sqrt_plus':
        return float(base + c * math.sqrt(u))
    if model['branch'] == 'sqrt_minus':
        return float(base - c * math.sqrt(u))
    return float(base - c * u)