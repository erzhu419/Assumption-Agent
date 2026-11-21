import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    m = np.array([r['m_2'] for r in train_rows], dtype=float)
    u = np.array([r['u_raw'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * m / (p[1] + m) + p[2] * np.sqrt(u) - y
    _params = least_squares(residual, [7.0, 1.3, 0.3], bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row):
    v, k, c = _params
    return float(v * row['m_2'] / (k + row['m_2']) + c * np.sqrt(row['u_raw']))