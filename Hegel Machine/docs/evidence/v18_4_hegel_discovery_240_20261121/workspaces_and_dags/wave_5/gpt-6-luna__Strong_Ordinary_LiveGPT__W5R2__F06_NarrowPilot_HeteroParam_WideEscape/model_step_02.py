import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Shared-gradient observations identify the saturating input curve; the
    # two deliberately varied gradients identify inhibition strength.
    def pred(p, r):
        vmax, rho = p
        u = float(r['u_now'])
        s = float(r['s_grad'])
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def residual(p):
        return [pred(p, r) - float(r['y']) for r in train_rows]
    res = least_squares(residual, x0=[6.0, 0.5], bounds=([0.0, 0.0], [100.0, 0.99]))
    return res.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = float(row['u_now']); s = float(row['s_grad'])
    return float(p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u))

_TRAIN_ROWS = []
# fit() stores parameters for predict() in the standard evaluation flow is unavailable;
# expose fit assignment-compatible prediction by using the supplied rows via module state.
def fit(train_rows):
    global _TRAIN_ROWS, _PARAMS
    _TRAIN_ROWS = list(train_rows)
    def f(p, r):
        v, rho = p; u=float(r['u_now']); s=float(r['s_grad'])
        return v*(1-rho*s/(1+s))*u/(1.2+u)
    z=least_squares(lambda p: [f(p,r)-float(r['y']) for r in _TRAIN_ROWS], [6.0,0.5], bounds=([0,0],[100,0.99]))
    _PARAMS=z.x
    return _PARAMS

def predict(row):
    v,rho=_PARAMS; u=float(row['u_now']); s=float(row['s_grad'])
    return float(v*(1-rho*s/(1+s))*u/(1.2+u))