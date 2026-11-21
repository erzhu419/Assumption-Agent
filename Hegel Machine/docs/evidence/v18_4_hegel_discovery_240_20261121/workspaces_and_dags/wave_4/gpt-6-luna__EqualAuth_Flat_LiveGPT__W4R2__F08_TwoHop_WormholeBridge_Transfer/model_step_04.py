import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # At the observed horizon and fixed state, model the response as a
    # saturating input contribution plus a smooth gradient-dependent term.
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p, uu, ss):
        a, b, c, d, e = p
        return a + b * (1.0 - np.exp(-np.maximum(c, 1e-8) * uu)) + d * ss * np.exp(-e * uu)
    def resid(p):
        return pred(p, u, s) - y
    p0 = np.array([0.0, 3.0, 0.8, 1.0, 0.3])
    result = least_squares(resid, p0, bounds=([-20,-20,1e-6,-20,0],[20,20,10,20,10]), max_nfev=20000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    a, b, c, d, e = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(a + b * (1.0 - math.exp(-max(c, 1e-8) * u)) + d * s * math.exp(-e * u))