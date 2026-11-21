import numpy as np
from scipy.optimize import least_squares

_params = None

def _flux(u, g, p):
    a, k, b, h, c, l, q = p
    return a*u/(k+u) + b*u/(h+u) + c*u/((l+u)*(1+q*g))

def fit(train_rows):
    global _params
    u, g, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _flux(u, g, p)-y,
                            [1.7, 0.7, 2.2, 3.8, 5.7, 1.2, 1.0],
                            bounds=(1e-8, np.inf), x_scale='jac',
                            ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return _params

def predict(row):
    return float(_flux(float(row['u_now']), float(row['s_grad']), _params))
