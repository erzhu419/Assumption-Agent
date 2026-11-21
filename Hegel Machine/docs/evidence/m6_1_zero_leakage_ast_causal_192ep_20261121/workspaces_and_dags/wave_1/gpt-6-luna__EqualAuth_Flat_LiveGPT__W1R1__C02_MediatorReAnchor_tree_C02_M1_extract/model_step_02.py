import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_raw'])
    s = float(row['s_surf'])
    return np.array([s, math.sqrt(max(u, 0.0)), float(row['m_1']), float(row['m_2']), float(row['m_3']), float(row['t_proc'])], dtype=float)

def _predict_params(p, row):
    vmax, km, c, b1, b2, b3, bt = p
    s = float(row['s_surf'])
    u = float(row['u_raw'])
    return vmax * s / (km + s) + c * math.sqrt(max(u, 0.0)) + b1 * float(row['m_1']) + b2 * float(row['m_2']) + b3 * float(row['m_3']) + bt * float(row['t_proc'])

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    # Fit the recalled cold-regime saturation/raw-input law together with additive
    # mediator and processing terms. Weak regularization stabilizes correlated cold rows.
    def residual(p):
        pred = np.asarray([_predict_params(p, r) for r in rows])
        penalty = np.array([0.0, 0.0, 0.0, 0.08*p[3], 0.08*p[4], 0.08*p[5], 0.08*p[6]])
        return np.concatenate([pred-y, penalty])
    p0 = np.array([7.0, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0])
    result = least_squares(residual, p0, bounds=([0.01, 0.01, -10, -10, -10, -10, -10], [30, 20, 10, 10, 10, 10, 10]), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    p = fit(row.get('train_rows', []))
    return float(_predict_params(p, row))