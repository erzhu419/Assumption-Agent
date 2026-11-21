import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_family = None

def _value(p, row, family):
    u = row['u_now']
    s = row['s_grad']
    if family == 'sqrt':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) + p[2] * math.sqrt(max(0.0, u))
    if family == 'linear':
        return p[0] * (1.0 - p[1] * s / (1.0 + s)) * u / (1.2 + u) - p[2] * u
    if family == 'hill':
        return p[0] * u*u / (p[1] + u*u) / (1.0 + p[2]*s) - p[3]*u
    return p[0]*u*math.exp(-p[1]/(0.5+u))*(1.0-p[2]*s/(1.0+s))+p[3]*math.sqrt(max(0.0,u))

def fit(train_rows):
    global _params, _family
    specs = [
        ('sqrt', [8.0, 0.7, 0.3], ([0.0, 0.0, -20.0], [100.0, 1.0, 20.0])),
        ('linear', [8.0, 0.7, 0.1], ([0.0, 0.0, -20.0], [100.0, 1.0, 20.0])),
        ('hill', [8.0, 1.0, 0.5, 0.1], ([0.0, 0.0001, 0.0, -20.0], [100.0, 100.0, 100.0, 20.0])),
        ('barrier', [4.0, 1.0, 0.7, 0.3], ([0.0, 0.0, 0.0, -20.0], [100.0, 20.0, 1.0, 20.0]))]
    best = float('inf')
    for family, initial, bounds in specs:
        result = least_squares(lambda p: np.array([_value(p, r, family)-r['y'] for r in train_rows]), initial, bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        score = float(np.dot(result.fun, result.fun))
        if score < best:
            best, _family, _params = score, family, result.x

def predict(row):
    return float(_value(_params, row, _family))