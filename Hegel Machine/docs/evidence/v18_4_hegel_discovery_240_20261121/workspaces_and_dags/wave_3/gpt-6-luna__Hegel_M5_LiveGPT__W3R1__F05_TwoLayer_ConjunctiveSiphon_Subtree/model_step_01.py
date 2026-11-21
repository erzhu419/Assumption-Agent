import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.05

def _features(row):
    u = max(float(row['u_now']) / TAU_SCALE, 0.0)
    s = float(row['s_grad'])
    base = (u / (1.2 + u))
    sink = s / (1.0 + s)
    root = math.sqrt(u)
    return base, sink, root

def _value(row, p):
    vmax, rho, c_loc = p
    base, sink, root = _features(row)
    return vmax * (1.0 - rho * sink) * base + c_loc * root

def fit(train_rows):
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    x0 = np.asarray([6.5, 0.70, 0.15], dtype=float)
    lo = np.asarray([3.836397, 0.658333, -0.822167], dtype=float)
    hi = np.asarray([9.163604, 0.748333, 1.108834], dtype=float)
    result = least_squares(
        lambda p: np.asarray([_value(r, p) for r in train_rows]) - ys,
        x0=np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi)
    )
    return result.x

def predict(row):
    return float(_value(row, fit([dict(row, y=row['y'])])))