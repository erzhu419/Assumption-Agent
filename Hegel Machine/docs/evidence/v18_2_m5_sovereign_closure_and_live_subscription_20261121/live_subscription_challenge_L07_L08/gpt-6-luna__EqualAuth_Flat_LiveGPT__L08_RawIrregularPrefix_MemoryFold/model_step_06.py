import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)

    def base(u, vmax, km):
        return vmax * u / (km + u)

    # Fit both archived signed-branch forms; retain the lower training-error fit.
    fits = []
    for kind in ('sqrt', 'leak'):
        def pred(p, r):
            vmax, km, branch = p
            u = max(float(r['u_now']), 0.0)
            q = math.sqrt(u) if kind == 'sqrt' else u
            return base(u, vmax, km) + branch * q
        def residual(p):
            return np.asarray([pred(p, r) for r in rows]) - y
        result = least_squares(residual, [2.0, 1.2, 0.0], bounds=([0.0, 1e-6, -10.0], [20.0, 20.0, 10.0]))
        fits.append((float(np.mean(residual(result.x)**2)), kind, result.x))
    _, kind, p = min(fits, key=lambda item: item[0])
    _params = (kind, np.asarray(p, dtype=float))
    return _params

def predict(row):
    if _params is None:
        raise ValueError('fit(train_rows) must be called before predict(row)')
    kind, p = _params
    vmax, km, branch = p
    u = max(float(row['u_now']), 0.0)
    q = math.sqrt(u) if kind == 'sqrt' else u
    return float(vmax * u / (km + u) + branch * q)
