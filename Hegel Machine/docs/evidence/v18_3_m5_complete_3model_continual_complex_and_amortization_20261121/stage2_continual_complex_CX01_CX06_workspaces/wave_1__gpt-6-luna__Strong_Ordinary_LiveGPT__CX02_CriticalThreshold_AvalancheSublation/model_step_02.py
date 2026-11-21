import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p, branch):
    vmax, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if branch == 'sqrt':
        return base + c * math.sqrt(max(u, 0.0))
    return base - c * u

def fit(train_rows):
    best = None
    for branch in ('sqrt', 'linear_leak'):
        def residual(p):
            return np.array([_value(r, p, branch) - float(r['y']) for r in train_rows])
        for start in ([4.0, 0.7, 0.2], [5.0, 0.5, 0.8], [3.0, 0.9, -0.2]):
            result = least_squares(residual, np.asarray(start, dtype=float), max_nfev=20000)
            score = float(np.sum(residual(result.x) ** 2))
            if best is None or score < best[0]:
                best = (score, branch, result.x.copy())
    return {'branch': best[1], 'params': best[2]}

def predict(row):
    # Parameters are supplied by fit() via the fitted model argument in the evaluator.
    raise RuntimeError('predict requires fitted parameters; use predict_model(model, row)')

def predict_model(model, row):
    return float(_value(row, model['params'], model['branch']))