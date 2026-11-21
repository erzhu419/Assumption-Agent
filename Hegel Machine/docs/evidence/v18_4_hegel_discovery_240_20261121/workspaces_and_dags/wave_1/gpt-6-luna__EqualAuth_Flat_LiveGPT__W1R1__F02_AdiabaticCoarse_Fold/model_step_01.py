import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        E, eta = np.exp(p)
        return (u / E) * (-np.expm1(-E * H / eta))
    scale = max(float(np.mean(np.abs(y))), 1.0)
    res = least_squares(lambda p: (pred(p) - y) / scale,
                        x0=np.log([0.48, 0.9]), max_nfev=20000)
    E, eta = np.exp(res.x)
    return {'E': float(E), 'eta': float(eta)}

def predict(row):
    E, eta = _params['E'], _params['eta']
    return float((row['u_now'] / E) * (-np.expm1(-E * row['H'] / eta)))

_params = {'E': 0.48, 'eta': 0.9}
_original_fit = fit
def fit(train_rows):
    global _params
    _params = _original_fit(train_rows)
    return _params
