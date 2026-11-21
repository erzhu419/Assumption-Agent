import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    a, b, c, d, e = p
    return a*u/(1.0 + b*u + c*s) + d*math.sqrt(u)/(1.0 + e*s)

def fit(train_rows):
    global _params
    rows = [(float(r['u_now']), float(r['s_grad']), float(r['y'])) for r in train_rows]
    def residuals(p):
        return np.array([_law(u, s, p) - y for u, s, y in rows])
    _params = least_squares(residuals, [7.0, 1.3, 0.6, 1.5, 2.5], bounds=(0.0, np.inf), gtol=1e-12).x

def predict(row):
    return float(_law(float(row['u_now']), float(row['s_grad']), _params))