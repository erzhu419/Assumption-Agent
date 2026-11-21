import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global _params
    rows = train_rows
    x = np.array([[float(r['m_2']), math.sqrt(max(0.0, float(r['u_raw'])))] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        vmax, km, c = p
        pred = vmax * x[:, 0] / (km + x[:, 0]) + c * x[:, 1]
        return pred - y
    result = least_squares(residual, x0=[8.0, 1.0, 0.5], bounds=([0.0, 1e-8, -10.0], [100.0, 100.0, 10.0]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    vmax, km, c = _params
    m2 = float(row['m_2'])
    u = max(0.0, float(row['u_raw']))
    return float(vmax * m2 / (km + m2) + c * math.sqrt(u))