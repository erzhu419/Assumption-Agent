import math
import numpy as np
from scipy.optimize import least_squares

_gain = None

def fit(train_rows):
    global _gain
    rows = list(train_rows)
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: p[0] * u - y, [1.0])
    _gain = float(result.x[0])
    return _gain

def predict(row) -> float:
    return float(_gain * float(row['u_now']))
