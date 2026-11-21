import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(row, p, kind):
    d = float(row['u_now']) - float(row['u_past'])
    h = float(row['H'])
    z = abs(d) / 0.004
    if kind == 0:
        response = p[0] * d / (1.0 + p[1] * z + p[2] * h)
    elif kind == 1:
        response = p[0] * d * np.exp(-p[2] * h) / (1.0 + p[1] * z)
    else:
        response = p[0] * d * np.exp(-p[2] * h) / (1.0 + p[1] * z + p[3] * h)
    return float(row['x_now']) + response

def fit(train_rows):
    global _model, _params
    best = None
    for kind in range(3):
        n = 4 if kind == 2 else 3
        for start in ([800.0, 0.05, 0.5, 0.2], [1000.0, 0.2, 1.0, 0.5], [700.0, 0.001, 0.6, 0.001]):
            result = least_squares(lambda p: np.array([_value(r, p, kind) - float(r['y']) for r in train_rows]), np.array(start[:n]), bounds=(np.zeros(n), np.full(n, np.inf)), max_nfev=5000)
            error = float(np.sum(result.fun ** 2))
            if best is None or error < best[0]:
                best = (error, kind, result.x)
    _, _model, _params = best

def predict(row):
    return float(_value(row, _params, _model))