import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_raw']) for r in train_rows])
    s = np.array([float(r['s_surf']) for r in train_rows])
    t = np.array([float(r['t_proc']) for r in train_rows])
    m = np.array([float(r['m_1']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def values(p, u, s, t, m):
        vmax, km, c, d1, d2, d3 = p
        return (vmax * s / (km + s) + c * np.sqrt(np.maximum(u, 0.0))
                - t * (d1 * m + d2 * m * m) - d3 * t * t * m)

    result = least_squares(
        lambda p: values(p, u, s, t, m) - y,
        x0=[8.0, 1.0, 0.5, 1.0, 0.1, 0.1],
        bounds=([0.0, 1e-8, -10.0, -30.0, -30.0, -30.0],
                [30.0, 30.0, 10.0, 30.0, 30.0, 30.0]),
        max_nfev=30000)
    _params = result.x
    return predict

def predict(row):
    vmax, km, c, d1, d2, d3 = _params
    u = max(float(row['u_raw']), 0.0)
    s = float(row['s_surf'])
    t = float(row['t_proc'])
    m = float(row['m_1'])
    return float(vmax * s / (km + s) + c * math.sqrt(u)
                 - t * (d1 * m + d2 * m * m) - d3 * t * t * m)
