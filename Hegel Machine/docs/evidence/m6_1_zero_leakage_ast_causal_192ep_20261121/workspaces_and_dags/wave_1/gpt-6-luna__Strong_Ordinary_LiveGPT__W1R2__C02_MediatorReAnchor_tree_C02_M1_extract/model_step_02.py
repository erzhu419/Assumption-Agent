import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row.get('u_raw', 0.0))
    t = float(row.get('t_proc', 0.0))
    s = float(row.get('s_surf', 0.0))
    m1 = float(row.get('m_1', 0.0))
    m2 = float(row.get('m_2', 0.0))
    m3 = float(row.get('m_3', 0.0))
    return np.array([1.0, math.sqrt(max(u, 0.0)), s, t, m1, m2, m3], dtype=float)


def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    # Fit a regularized additive response: retain nonlinear cold-regime substrate
    # saturation, while allowing independent effects after mediator clamping.
    def predict_params(p, row):
        u = max(float(row.get('u_raw', 0.0)), 0.0)
        s = max(float(row.get('s_surf', 0.0)), 0.0)
        t = float(row.get('t_proc', 0.0))
        m1 = float(row.get('m_1', 0.0)); m2 = float(row.get('m_2', 0.0)); m3 = float(row.get('m_3', 0.0))
        vmax, k, c_raw, c_t, b1, b2, b3, intercept = p
        return vmax * s / (k + s + 1e-9) + c_raw * math.sqrt(u) + c_t * t + b1*m1 + b2*m2 + b3*m3 + intercept
    def residual(p):
        return np.asarray([predict_params(p, r) for r in train_rows]) - y
    p0 = np.array([8.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = least_squares(residual, p0, bounds=([-50, 1e-4, -20, -20, -20, -20, -20, -20], [50, 100, 20, 20, 20, 20, 20, 20]), max_nfev=5000)
    return result.x.tolist()


def predict(row):
    p = fit([row])
    return float(p[0] * max(float(row.get('s_surf', 0.0)), 0.0) / (p[1] + max(float(row.get('s_surf', 0.0)), 0.0) + 1e-9) + p[2] * math.sqrt(max(float(row.get('u_raw', 0.0)), 0.0) + 1e-12) + p[3] * float(row.get('t_proc', 0.0)) + p[4] * float(row.get('m_1', 0.0)) + p[5] * float(row.get('m_2', 0.0)) + p[6] * float(row.get('m_3', 0.0)) + p[7])