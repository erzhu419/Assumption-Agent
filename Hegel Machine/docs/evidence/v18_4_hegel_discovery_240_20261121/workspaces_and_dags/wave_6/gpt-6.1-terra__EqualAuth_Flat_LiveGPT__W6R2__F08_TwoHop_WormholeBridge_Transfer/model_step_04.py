import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _eval(row, p, branch):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u)
    return base + p[2] * (math.sqrt(max(0.0, u)) if branch == 'sqrt' else u)

def fit(train_rows):
    global _model, _params
    y = np.array([float(r['y']) for r in train_rows])
    candidates = []
    for branch in ('sqrt', 'linear'):
        result = least_squares(lambda p: np.array([_eval(r, p, branch) for r in train_rows]) - y, [5.0, 0.7, 0.0], bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]))
        candidates.append((float(np.sum(result.fun ** 2)), branch, result.x))
    _, _model, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_eval(row, _params, _model))