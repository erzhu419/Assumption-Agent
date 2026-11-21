import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-9, float(r['u_now'])) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    base = lambda vmax, rho: vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    candidates = []
    # Test the two documented open-branch alternatives, fitting all parameters numerically.
    for branch in ('sqrt', 'leak'):
        def pred(p):
            vmax, rho, c = p
            return base(vmax, rho) + (c * np.sqrt(u) if branch == 'sqrt' else -c * u)
        for seed in ([3.0, 0.5, 1.0], [5.0, 0.2, 0.2], [2.0, 0.8, 2.0]):
            res = least_squares(lambda p: pred(p) - y, seed, bounds=([0, 0, 0], [100, 0.999, 100]), max_nfev=10000)
            candidates.append((float(np.mean((pred(res.x)-y)**2)), branch, res.x))
    _, branch, params = min(candidates, key=lambda z: z[0])
    return (branch, params.tolist())

def predict(row):
    branch, p = fit(_TRAIN_ROWS)
    vmax, rho, c = p
    u = max(1e-9, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    val = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    val += c * math.sqrt(u) if branch == 'sqrt' else -c * u
    return float(val)