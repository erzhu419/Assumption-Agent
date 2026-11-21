import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        out = []
        for r in rows:
            h = float(r['H'])
            d = float(r['u_now']) - float(r['u_past'])
            log_gain = p[0] + p[1]*h + p[2]*h*h + p[3]*h*h*h
            out.append(float(r['x_now']) + d*np.exp(log_gain) - float(r['y']))
        return np.asarray(out)
    result = least_squares(residual, np.array([1.0, -0.75, 0.08, 0.0]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    h = float(row['H'])
    d = float(row['u_now']) - float(row['u_past'])
    p = _params
    return float(float(row['x_now']) + d*np.exp(p[0] + p[1]*h + p[2]*h*h + p[3]*h*h*h))