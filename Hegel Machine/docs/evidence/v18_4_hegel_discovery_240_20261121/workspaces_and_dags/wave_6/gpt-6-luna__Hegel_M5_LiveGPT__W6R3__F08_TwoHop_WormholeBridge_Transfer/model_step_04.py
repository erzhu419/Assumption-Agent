import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_sqrt = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_sqrt * math.sqrt(u)

def fit(train_rows):
    rows = [r for r in train_rows if isinstance(r, dict) and 'y' in r]
    if not rows:
        raise ValueError('fit requires rows containing y')
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.7, 0.2]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    return result.x

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else row.get('params')
    if p is None:
        raise ValueError('predict requires fitted params or train_rows')
    return float(_value(row, p))