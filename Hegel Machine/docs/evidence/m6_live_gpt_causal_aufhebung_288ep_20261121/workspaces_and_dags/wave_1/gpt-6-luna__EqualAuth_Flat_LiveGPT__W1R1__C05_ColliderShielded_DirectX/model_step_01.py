import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = [r for r in train_rows if all(k in r for k in ('x_ind', 'u_ctrl', 'y'))]
    if not rows:
        return {'vmax': 0.0, 'c_ctrl': 0.0}
    x = np.asarray([float(r['x_ind']) for r in rows])
    u = np.asarray([float(r['u_ctrl']) for r in rows])
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([7.0, 1.0]))
    return {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}

def predict(row):
    p = fit([row]) if not hasattr(predict, '_params') else predict._params
    return float(p['vmax'] * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + p['c_ctrl'] * float(row['u_ctrl']))
