import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _f(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(0.0, float(row['u_now']) / 3.0)
    s = max(0.0, float(row['s_grad']))
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_f(r, p) for r in rows]) - y
    best = None
    for start in ([5.0, 1.0, 0.5, 0.0], [10.0, 2.0, 0.7, 0.2], [5.0, 0.2, 0.2, -0.2]):
        sol = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.0, -5.0, -10.0], [100.0, 20.0, 5.0, 10.0]), max_nfev=20000)
        score = float(np.dot(sol.fun, sol.fun))
        if best is None or score < best[0]:
            best = (score, sol.x)
    _P = np.asarray(best[1], dtype=float)
    return _P.copy()

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_f(row, _P))