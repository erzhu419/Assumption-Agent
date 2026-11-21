import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)

    def predict_with(p, row):
        vmax, km, c_raw, d_thermal = p
        m2 = max(0.0, float(row.get('m_2', 0.0)))
        u = max(0.0, float(row.get('u_raw', 0.0)))
        t = float(row.get('t_proc', 0.0))
        return vmax * m2 / (km + m2 + 1e-12) + c_raw * math.sqrt(u) - d_thermal * t

    def residual(p):
        return np.array([predict_with(p, row) - float(row['y']) for row in rows])

    starts = ([8.0, 1.0, 0.5, 1.0], [10.0, 2.0, 0.1, 2.0], [5.0, 0.5, 1.0, 0.5])
    results = [least_squares(residual, x0, bounds=([0.0, 1e-5, -10.0, -20.0], [100.0, 100.0, 10.0, 20.0])) for x0 in starts]
    _PARAMS = min(results, key=lambda r: float(np.dot(r.fun, r.fun))).x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    vmax, km, c_raw, d_thermal = _PARAMS
    m2 = max(0.0, float(row.get('m_2', 0.0)))
    u = max(0.0, float(row.get('u_raw', 0.0)))
    t = float(row.get('t_proc', 0.0))
    return float(vmax * m2 / (km + m2 + 1e-12) + c_raw * math.sqrt(u) - d_thermal * t)