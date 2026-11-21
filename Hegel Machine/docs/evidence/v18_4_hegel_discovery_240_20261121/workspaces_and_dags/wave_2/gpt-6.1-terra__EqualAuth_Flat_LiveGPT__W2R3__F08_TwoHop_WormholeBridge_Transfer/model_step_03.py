import numpy as np
from scipy.optimize import least_squares

_params = None

def _formula(p, u, s):
    # Saturating supply response with a gradient-dependent transfer term.
    a, b, c, d, e, f = p
    return a + b*u/(np.exp(c)+u) + (d + e*u/(np.exp(c)+u))/(np.exp(f)+s)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for c in [-1.0, 0.0, 1.0]:
        for f in [-1.0, 0.0, 1.0]:
            p0 = [0.0, 3.0, c, 1.0, 0.0, f]
            res = least_squares(lambda p: _formula(p, u, s)-y, p0,
                                bounds=([-20,-20,-6,-20,-20,-6],
                                        [20,20,6,20,20,6]),
                                max_nfev=5000)
            if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
                best = res
    _params = best.x

def predict(row):
    return float(_formula(_params, float(row['u_now']), float(row['s_grad'])))