import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

# Fit the verified stationary siphon candidates. A signed square-root branch
# term accommodates either open-branch direction; the barrier variant is
# retained because the pilot does not resolve it from the base siphon.
def _c(row, p):
    vmax, rho, c = p
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c * math.sqrt(u)

def _r(row, p):
    j0, beta, rho, c = p
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def _q(row, p):
    vmax, k_half, alpha, leak = p
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha*s) - leak*u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    candidates = [
        ('C', _c, [6.0, 0.5, 0.5], ([0.0, -2.0, -10.0], [30.0, 3.0, 10.0])),
        ('R', _r, [5.0, 0.2, 0.5, 0.5], ([0.0, 0.0, -2.0, -10.0], [50.0, 10.0, 3.0, 10.0])),
        ('Q', _q, [8.0, 2.0, 0.2, 0.1], ([0.0, 1e-6, 0.0, -10.0], [50.0, 100.0, 10.0, 10.0])),
    ]
    best = None
    for name, fn, x0, bounds in candidates:
        def residual(p):
            return np.asarray([fn(r, p) for r in rows]) - y
        result = least_squares(residual, x0=np.asarray(x0), bounds=bounds, max_nfev=30000)
        rss = float(np.dot(result.fun, result.fun))
        if best is None or rss < best[0]:
            best = (rss, name, result.x)
    _PARAMS = (best[1], np.asarray(best[2], dtype=float))
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    name, p = _PARAMS
    return float({'C': _c, 'R': _r, 'Q': _q}[name](row, p))
