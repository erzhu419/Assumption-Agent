import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, sink, branch, loss, threshold = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (vmax - sink * s / (1.0 + s)) * u / (1.2 + u) + branch * np.sqrt(u) - loss * u * max(s - threshold, 0.0)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [7.0, 4.8, 0.35, 1.4, 1.35], bounds=([0, 0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_model(row, _params))