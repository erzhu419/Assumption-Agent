import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def features(r):
        un = float(r['u_now']) / 4.0
        s = float(r['s_grad'])
        base = un / (1.2 + un)
        inhib = s / (1.0 + s)
        return base, inhib, np.sqrt(max(un, 0.0))
    X = [features(r) for r in rows]
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        vmax, rho, c = p
        return np.array([vmax * b * (1.0 - rho * h) + c * q for b, h, q in X]) - y
    best = None
    for start in ([4.0, 0.7, 0.0], [4.0, 0.3, 1.0], [4.0, 1.2, -1.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, -np.inf], [np.inf, 3.0, np.inf]))
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return tuple(float(v) for v in best.x)

def predict(row):
    vmax, rho, c = fit([row]) if False else _params
    un = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c * np.sqrt(max(un, 0.0)))
