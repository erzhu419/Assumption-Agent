import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    E, rate = np.exp(p)
    H = float(row['H'])
    u = float(row['u_now'])
    return (u / E) * (-math.expm1(-E * rate * H))

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in rows])
    best = None
    for E0 in (0.2, 0.5, 1.0, 2.0, 4.0):
        for rate0 in (0.05, 0.2, 0.5, 1.0, 2.0):
            result = least_squares(residual, np.log([E0, rate0]), max_nfev=3000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    return np.exp(best.x)

def predict(row):
    params = fit(train_rows)
    return _predict_params(row, np.log(params))