import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def predict_p(p, r):
        a, b, c, d = p
        h = float(r['H'])
        x = float(r['x_now'])
        u = float(r['u_now'])
        up = float(r['u_past'])
        # Relaxation toward an input-dependent level, with a residual history contrast.
        return (x * math.exp(-a*h) + (b*u + c*up + d) * (1.0-math.exp(-a*h)))
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([predict_p(p, r) for r in rows]) - y
    best = None
    for start in ([1.0, 0.5, 0.0, 0.0], [2.0, 0.2, 0.2, 0.2], [0.5, 1.0, -0.2, 0.0]):
        res = least_squares(residual, np.array(start, dtype=float), max_nfev=3000)
        score = float(np.mean(res.fun**2))
        if best is None or score < best[0]:
            best = (score, res.x)
    params = best[1].tolist()
    def model(row):
        return float(predict_p(params, row))
    return model

def predict(row):
    return fit([])(row)