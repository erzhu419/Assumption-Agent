import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    a = np.array([[r['u_now'], r['s_grad'], r['y']] for r in train_rows], dtype=float)
    u, g, y = a.T
    # Smooth gradient-dependent saturating response plus offset.
    def pred(p, u, g):
        b, k, c, d, e = p
        return c + (b + d*g) * u / (k + u) + e*g
    def residual(p):
        return pred(p, u, g) - y
    p0 = np.array([8., 1., 0., -1., 0.])
    res = least_squares(residual, p0, max_nfev=20000)
    return res.x.tolist()

def predict(row):
    b, k, c, d, e = fit(_train_rows)
    return float(c + (b + d*row['s_grad']) * row['u_now'] / (k + row['u_now']) + e*row['s_grad'])