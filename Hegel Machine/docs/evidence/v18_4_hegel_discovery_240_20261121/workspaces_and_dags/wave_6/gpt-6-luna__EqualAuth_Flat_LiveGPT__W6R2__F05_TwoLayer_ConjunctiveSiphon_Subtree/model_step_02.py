import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def _model(p, row):
    u, s = _features(row)
    # Concave saturating input response with a smooth, bounded sink-gradient penalty.
    a, b, rho, q = p
    return a + b * math.sqrt(u) - rho * (s / (1.0 + s)) * (u ** q)

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([0.0, 8.0, 1.0, 0.5]),
                           bounds=([-100.0, -100.0, -100.0, 0.05],
                                   [100.0, 100.0, 100.0, 3.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_model(p, row))