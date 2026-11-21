import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    zs = np.array([[float(r['z_%d' % j]) for j in range(1, 5)] for r in train_rows])
    best = None
    # Search thresholded indicator branches suggested by the tree-A signature.
    for j in range(4):
        vals = np.unique(zs[:, j])
        thresholds = [(vals[k] + vals[k+1]) / 2.0 for k in range(len(vals)-1)]
        if not thresholds:
            thresholds = [float(np.median(vals))]
        for threshold in thresholds:
            ind = (zs[:, j] > threshold).astype(float)
            for power in (1.0, 2.0):
                q = np.maximum(x, 0.0) ** power
                def residual(p):
                    vmax, half, c, branch = p
                    return vmax*q/(half+q+1e-12) - c*u + branch*ind - y
                res = least_squares(residual, [8.0, 2.0, 0.1, 1.0], bounds=([-100,1e-8,-100,-100],[100,100,100,100]))
                score = float(np.mean(res.fun**2))
                if best is None or score < best[0]:
                    best = (score, j, float(threshold), power, tuple(float(v) for v in res.x))
    _, j, threshold, power, params = best
    _PARAMS = (j, threshold, power, params)
    return _PARAMS

def predict(row):
    j, threshold, power, p = _PARAMS
    vmax, half, c, branch = p
    x = max(float(row['x_ind']), 0.0)
    q = x ** power
    ind = 1.0 if float(row['z_%d' % (j+1)]) > threshold else 0.0
    return float(vmax*q/(half+q+1e-12) - c*float(row['u_ctrl']) + branch*ind)
