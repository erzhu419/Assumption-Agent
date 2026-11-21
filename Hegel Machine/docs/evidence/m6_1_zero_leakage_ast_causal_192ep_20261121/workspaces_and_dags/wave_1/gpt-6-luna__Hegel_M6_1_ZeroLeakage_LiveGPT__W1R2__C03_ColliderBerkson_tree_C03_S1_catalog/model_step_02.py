import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Fit y = a1*log(1+u1) + a2*u2/(1.5+u2); this uses independently varying
    # u1/u2 values and does not condition deployment on archive selection.
    X = np.array([[math.log1p(float(r['u_1'])), float(r['u_2'])/(1.5+float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 3.0]))
    return tuple(float(v) for v in res.x)

def predict(row):
    a1, a2 = fit(_TRAIN_ROWS)
    u1, u2 = float(row['u_1']), float(row['u_2'])
    return float(a1*math.log1p(u1) + a2*u2/(1.5+u2))
