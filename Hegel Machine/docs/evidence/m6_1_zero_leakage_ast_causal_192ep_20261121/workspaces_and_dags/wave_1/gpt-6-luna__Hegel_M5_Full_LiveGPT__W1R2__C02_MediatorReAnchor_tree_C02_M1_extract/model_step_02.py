import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Fit the recalled cold-regime saturating law using m_2 as the mediator,
    # with a linear processing-time term for thermal deployment.
    u = np.array([float(r['u_raw']) for r in train_rows])
    t = np.array([float(r['t_proc']) for r in train_rows])
    m = np.array([float(r['m_2']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, k, c, beta = p
        return vmax * m / (k + m) + c * np.sqrt(np.maximum(u, 0.0)) + beta * t
    result = least_squares(lambda p: pred(p) - y,
                           x0=np.array([8.0, 2.0, 0.5, 0.0]),
                           bounds=([0.0, 1e-8, -20.0, -20.0], [100.0, 100.0, 20.0, 20.0]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    vmax, k, c, beta = p
    u = max(float(row['u_raw']), 0.0)
    m = float(row['m_2'])
    t = float(row['t_proc'])
    return float(vmax * m / (k + m) + c * math.sqrt(u) + beta * t)
