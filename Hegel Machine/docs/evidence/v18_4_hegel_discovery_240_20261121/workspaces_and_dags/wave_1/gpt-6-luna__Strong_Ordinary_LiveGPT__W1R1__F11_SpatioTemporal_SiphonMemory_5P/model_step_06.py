import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    x = float(row['x_now'])
    u = float(row['u_now'])
    m = float(row['mem_conv'])
    h = float(row['H'])
    # Two relaxing modes: initial state and memory, plus driven approach to equilibrium.
    a, b, cx, cm, cu, ceq = p
    return cx*x*np.exp(-a*h) + cm*m*np.exp(-b*h) + cu*u + ceq*u*(1.0-np.exp(-b*h))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    best = None
    for start in ([1, 2, .4, .2, .2, .3], [.5, 4, .5, .5, .1, .2], [3, .5, .3, .1, .3, .1]):
        result = least_squares(residual, np.array(start, dtype=float), bounds=([0.001,0.001,-10,-10,-10,-10],[20,20,10,10,10,10]), max_nfev=10000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(_params, row))