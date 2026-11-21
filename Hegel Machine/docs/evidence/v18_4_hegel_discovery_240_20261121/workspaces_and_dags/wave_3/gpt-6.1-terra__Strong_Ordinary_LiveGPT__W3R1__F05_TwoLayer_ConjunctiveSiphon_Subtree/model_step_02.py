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
    sink = base * s / (1.0 + s)
    candidates = []
    for name, branch in [('sqrt', np.sqrt(u)), ('linear', u)]:
        X = np.column_stack((base, -sink, branch))
        result = least_squares(lambda p: X @ p - y, [5.0, 3.5, 0.5])
        candidates.append((float(np.sum(result.fun ** 2)), name, result.x))
    _, _model, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    u = row['u_now'] / 0.05
    s = row['s_grad']
    branch = math.sqrt(u) if _model == 'sqrt' else u
    return float(_params[0] * u / (1.2 + u) - _params[1] * u / (1.2 + u) * s / (1.0 + s) + _params[2] * branch)
