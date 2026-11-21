import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _rational(p, u, s):
    num = p[0] + p[1]*u + p[2]*s + p[3]*u*s
    den = 1.0 + p[4]*u + p[5]*s + p[6]*u*s
    return num / den

def _additive(p, u, s):
    return p[0] + p[1]*u/(1.0 + p[2]*u) + p[3]/(1.0 + p[4]*s)

def fit(train_rows):
    global _model, _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    for start in ([0.1, 2.0, 0.2, 2.0, 0.5], [0.0, 4.0, 1.0, 2.0, 1.0]):
        result = least_squares(lambda p: _additive(p, u, s) - y, start,
                               bounds=([-20, -20, 0, -20, 0], [20, 20, 20, 20, 20]), max_nfev=10000)
        candidates.append((np.mean(result.fun**2), _additive, result.x))
    for start in ([0.1, 3.0, 0.1, 0.0, 0.5, 0.5, 0.0], [0.5, 4.0, 0.0, 0.0, 1.0, 1.0, 0.1]):
        result = least_squares(lambda p: _rational(p, u, s) - y, start,
                               bounds=([-20, -20, -20, -20, 0, 0, 0], [20, 20, 20, 20, 20, 20, 20]), max_nfev=10000)
        candidates.append((np.mean(result.fun**2), _rational, result.x))
    _, _model, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))
