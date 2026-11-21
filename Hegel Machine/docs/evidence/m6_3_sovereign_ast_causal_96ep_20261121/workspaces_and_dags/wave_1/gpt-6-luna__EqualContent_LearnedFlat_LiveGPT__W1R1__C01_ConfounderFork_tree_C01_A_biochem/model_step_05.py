import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.array([float(r['x_ind']) for r in train_rows])
    z = np.array([float(r['z_3']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    q = (np.maximum(x, 0.0) / 1.5) ** 2
    vals = np.unique(z)
    thresholds = [(vals[i] + vals[i+1]) / 2.0 for i in range(len(vals)-1)]
    best = None
    for t in thresholds:
        gate = (z > t).astype(float)
        # Tree branch changes the saturating flux, while control has an additive effect.
        def residual(p):
            vmax0, vmax1, half, c = p
            vmax = vmax0 + vmax1 * gate
            return vmax*q/(half+q+1e-12) - c*u - y
        for start in ([8.0, 0.0, 1.0, 0.1], [3.0, 8.0, 2.0, 0.0], [12.0, -5.0, 0.5, 0.5]):
            res = least_squares(residual, start, bounds=([-100,-100,1e-8,-100],[100,100,100,100]), max_nfev=5000)
            score = float(np.mean(res.fun ** 2))
            if best is None or score < best[0]:
                best = (score, float(t), tuple(float(v) for v in res.x))
    _, threshold, params = best
    _PARAMS = (threshold, params)
    return _PARAMS

def predict(row):
    threshold, p = _PARAMS
    vmax0, vmax1, half, c = p
    x = max(float(row['x_ind']), 0.0)
    q = (x / 1.5) ** 2
    gate = 1.0 if float(row['z_3']) > threshold else 0.0
    return float((vmax0 + vmax1*gate)*q/(half+q+1e-12) - c*float(row['u_ctrl']))
