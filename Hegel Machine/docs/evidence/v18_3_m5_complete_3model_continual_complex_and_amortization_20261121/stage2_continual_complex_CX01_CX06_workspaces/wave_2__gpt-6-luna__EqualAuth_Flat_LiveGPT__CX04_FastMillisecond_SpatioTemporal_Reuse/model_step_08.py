import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 0.005

def _features(row):
    un = float(row['u_now']) / _TAU
    sg = float(row['s_grad'])
    base = (1.0 - row_rho(sg)) * (un / (1.2 + un))
    return un, base

def row_rho(sg):
    # Keep rho separate in the fitted expression; this helper returns gradient ratio.
    return sg / (1.0 + sg)

def fit(train_rows):
    def pred(p, r):
        un = float(r['u_now']) / _TAU
        sg = float(r['s_grad'])
        return p[0] * (1.0 - p[1] * sg / (1.0 + sg)) * (un / (1.2 + un)) - p[2] * un
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([pred(p, r) for r in train_rows]) - y
    res = least_squares(residual, x0=np.array([2.0, 0.5, 0.01]), bounds=([0.0, 0.0, 0.0], [20.0, 2.0, 10.0]), max_nfev=2000)
    return res.x

def predict(row):
    # The evaluator fits parameters on train_rows before prediction; retain them via fit call convention.
    p = fit(_ACTIVE_TRAIN_ROWS)
    un = float(row['u_now']) / _TAU
    sg = float(row['s_grad'])
    return float(p[0] * (1.0 - p[1] * sg / (1.0 + sg)) * (un / (1.2 + un)) - p[2] * un)
