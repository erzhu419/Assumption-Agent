import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    z2 = np.array([float(r['z_2']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    gate = (z2 > 1.0).astype(float)
    # Hill saturation in the interventional input, with a z2 tree-indicator
    # contribution and a linear control penalty.
    def pred(p):
        vmax, k, c, gate_amp = p
        h = x*x / (k + x*x)
        return vmax*h - c*u + gate_amp*gate
    p0 = np.array([8.0, 1.5, 0.5, 0.0])
    res = least_squares(lambda p: pred(p)-y, p0,
                        bounds=([0.0, 1e-8, -10.0, -10.0], [100.0, 100.0, 10.0, 10.0]))
    _params = res.x
    return _params


def predict(row):
    vmax, k, c, gate_amp = _params
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    gate = 1.0 if float(row['z_2']) > 1.0 else 0.0
    return float(vmax * x*x/(k+x*x) - c*u + gate_amp*gate)
