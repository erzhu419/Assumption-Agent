import math
import numpy as np
from scipy.optimize import least_squares

def _gate(row):
    # Tree-A indicator inferred from the do-clamp rows: both elevated z1
    # and low z3 mark the two elevated-response cases.
    return float(row.get('z_1', 0.0) > 2.0 and row.get('z_3', 0.0) < 1.0)

def _features(row, p):
    vmax, k_half, c_ctrl, delta = p
    x = max(0.0, float(row.get('x_ind', 0.0))) / 1.5
    hill = x*x / (k_half + x*x)
    return vmax * hill - c_ctrl * float(row.get('u_ctrl', 0.0)) + delta * _gate(row)

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_features(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([8.0, 0.7, 0.5, 2.0]),
                           bounds=([0.01, 0.001, -5.0, -20.0], [30.0, 30.0, 5.0, 20.0]))
    return result.x

def predict(row):
    # fit(train_rows) returns the parameter vector; pass it as row['_params'].
    return float(_features(row, row['_params']))