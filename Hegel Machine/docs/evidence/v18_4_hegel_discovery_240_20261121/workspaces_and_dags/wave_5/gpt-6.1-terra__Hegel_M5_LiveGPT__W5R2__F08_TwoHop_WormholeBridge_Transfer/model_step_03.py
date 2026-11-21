import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    numerator = p[0] + p[1]*u + p[2]*s + p[3]*u*s
    denominator = 1.0 + p[4]*u + p[5]*s + p[6]*u*s + p[7]*s*s
    return numerator / denominator

def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - targets
    lower = [-np.inf]*4 + [0.0]*4
    upper = [np.inf]*8
    starts = [
        [0.0, 4.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.1],
        [0.2, 3.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0]
    ]
    solutions = [least_squares(residual, p, bounds=(lower, upper), max_nfev=10000) for p in starts]
    _params = min(solutions, key=lambda result: np.sum(result.fun**2)).x

def predict(row):
    return float(_value(_params, row))
