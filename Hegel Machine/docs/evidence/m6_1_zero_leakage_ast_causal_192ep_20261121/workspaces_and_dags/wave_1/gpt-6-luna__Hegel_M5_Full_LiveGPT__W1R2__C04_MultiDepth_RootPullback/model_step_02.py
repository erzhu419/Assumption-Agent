import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    global _params
    rows = list(train_rows)
    E = np.array([max(1e-12, float(r['u_indoor']) + float(r['u_outdoor']) * float(r['p_weather'])) for r in rows])
    impact = np.array([float(r['impact_idx']) * float(r['a_foot']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # Retain the recalled saturating effective-dose mechanism; allow a mild
    # dose exponent to capture possible nonlinear indoor contribution.
    def residual(p):
        g, k, loss, alpha = p
        dose = E ** alpha
        return g * dose / (k + dose) - loss * impact - y
    p0 = np.array([7.0, 1.0, 3.0, 1.0])
    result = least_squares(residual, p0, bounds=([0.01, 1e-5, 0.0, 0.25], [100.0, 100.0, 30.0, 2.5]), max_nfev=30000)
    _params = result.x
    return _params

def predict(row):
    g, k, loss, alpha = _params
    dose = max(1e-12, float(row['u_indoor']) + float(row['u_outdoor']) * float(row['p_weather'])) ** alpha
    return float(g * dose / (k + dose) - loss * float(row['impact_idx']) * float(row['a_foot']))