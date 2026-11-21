import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_raw']) for r in train_rows])
    m = np.array([float(r['m_2']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    X = np.column_stack([m, np.sqrt(np.maximum(u, 0.0))])
    # y = vmax*m/(k_m+m) + c_raw*sqrt(u)
    def residual(p):
        vmax, k_m, c_raw = p
        return vmax * m / (k_m + m) + c_raw * np.sqrt(np.maximum(u, 0.0)) - y
    result = least_squares(residual, x0=[6.0, 1.0, 0.5], bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    return {'vmax': float(result.x[0]), 'k_m': float(result.x[1]), 'c_raw': float(result.x[2])}

def predict(row):
    p = fit([row])
    m = float(row['m_2'])
    u = max(float(row['u_raw']), 0.0)
    return float(p['vmax'] * m / (p['k_m'] + m) + p['c_raw'] * np.sqrt(u))