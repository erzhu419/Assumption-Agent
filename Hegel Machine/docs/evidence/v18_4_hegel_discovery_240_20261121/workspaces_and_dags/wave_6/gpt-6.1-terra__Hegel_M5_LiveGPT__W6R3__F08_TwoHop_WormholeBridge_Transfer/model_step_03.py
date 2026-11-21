import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (p[0] + p[1]*u + p[2]*s + p[3]*u*s) / (1.0 + p[4]*u + p[5]*s + p[6]*u*s)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_model(p, r) - float(r['y']) for r in rows])
    result = least_squares(residual, [0.0, 5.0, 0.0, 1.0, 1.0, 0.5, 1.0], bounds=([-np.inf]*4 + [0.0]*3, [np.inf]*7), max_nfev=20000)
    _params = result.x

def predict(row):
    return float(_model(_params, row))