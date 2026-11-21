import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.02

def _value(p, row):
    c_x, gain, log_tau = p
    tau = math.exp(log_tau)
    delta = (float(row['u_now']) - float(row['u_past'])) / TAU_SCALE
    return c_x * float(row['x_now']) + gain * delta * math.exp(-float(row['H']) / tau)

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(p, r) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([1.0, 1.0, math.log(0.5)]), bounds=([-10.0, -10.0, math.log(1e-4)], [10.0, 10.0, math.log(100.0)]))
    return result.x

def predict(row):
    if 'params' in row:
        p = np.asarray(row['params'], dtype=float)
    else:
        raise ValueError("predict(row) requires fitted parameters in row['params']")
    return float(_value(p, row))