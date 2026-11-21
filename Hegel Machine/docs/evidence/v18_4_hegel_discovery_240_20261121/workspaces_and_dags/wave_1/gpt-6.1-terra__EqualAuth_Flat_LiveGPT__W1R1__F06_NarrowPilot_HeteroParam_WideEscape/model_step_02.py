import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    a, k, c, b = p
    u = float(row['u_now'])
    s = max(0.0, float(row['s_grad']))
    return a * u / (k + u) / (1.0 + c * s ** b)

def fit(train_rows):
    global _params
    def residual(logp):
        p = np.exp(logp)
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    best = None
    for start in ([5.5, 1.2, 1.0, 1.0], [8.0, 1.2, 2.0, 1.5], [4.0, 1.0, 0.5, 2.0]):
        result = least_squares(residual, np.log(start), bounds=(np.log([0.01, 0.01, 0.001, 0.1]), np.log([100.0, 100.0, 100.0, 5.0])), max_nfev=3000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = np.exp(best.x)

def predict(row):
    return float(_model(row, _params))