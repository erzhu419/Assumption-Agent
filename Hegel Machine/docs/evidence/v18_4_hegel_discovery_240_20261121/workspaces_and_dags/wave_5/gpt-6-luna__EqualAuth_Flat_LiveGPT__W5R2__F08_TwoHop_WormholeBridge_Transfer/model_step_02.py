import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X = np.array([[r['u_now'], r['s_grad']] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    # Saturating transfer in input, with linear and quadratic gradient modulation.
    def pred(p, x):
        a, k, b, c, d = p
        u, s = x
        return a * (1.0 - np.exp(-k * u)) + b*s + c*s*s + d*u*s
    def residual(p):
        return np.array([pred(p, x) for x in X]) - y
    best = None
    for init in ([3.5, 1.0, 1.0, -0.2, 0.0], [4.0, 0.5, 0.0, 0.0, 0.0], [3.0, 1.5, 2.0, -0.5, 0.0]):
        sol = least_squares(residual, init, bounds=([-100, 1e-5, -100, -100, -100], [100, 20, 100, 100, 100]), max_nfev=20000)
        if best is None or np.sum(sol.fun**2) < np.sum(best.fun**2):
            best = sol
    return best.x

def predict(row):
    p = fit(_train_rows)
    u, s = row['u_now'], row['s_grad']
    a, k, b, c, d = p
    return float(a * (1.0 - math.exp(-k * u)) + b*s + c*s*s + d*u*s)
