import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def fit(train_rows):
    global _model, _params
    u = np.array([r['u_now'] / 0.05 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    candidates = []
    for name, branch in [('sqrt', np.sqrt(u)), ('linear', u)]:
        def residual(p):
            return p[0] * base * (1.0 - p[1] * sink) + p[2] * branch - y
        result = least_squares(residual, [6.0, 0.7, 0.5], max_nfev=10000)
        candidates.append((float(np.dot(result.fun, result.fun)), name, result.x))
    _, _model, _params = min(candidates, key=lambda c: c[0])

def predict(row):
    u = row['u_now'] / 0.05
    s = row['s_grad']
    branch = math.sqrt(u) if _model == 'sqrt' else u
    return float(_params[0] * (1.0 - _params[1] * s / (1.0 + s)) * u / (1.2 + u) + _params[2] * branch)
