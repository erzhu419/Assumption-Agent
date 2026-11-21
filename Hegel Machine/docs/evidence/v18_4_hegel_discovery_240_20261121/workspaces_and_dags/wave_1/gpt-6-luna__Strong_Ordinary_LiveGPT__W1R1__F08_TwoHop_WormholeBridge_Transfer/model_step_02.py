import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def _value(row, p, branch):
    vmax, rho, c = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    if branch == 'sqrt':
        return base + c * math.sqrt(u)
    return base - c * u

def fit(train_rows):
    global _params, _branch
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    for branch in ('sqrt', 'leak'):
        def residual(p):
            return np.asarray([_value(r, p, branch) for r in train_rows]) - y
        # Positive vmax and rho, bounded to physically meaningful inhibition.
        result = least_squares(residual, x0=[6.0, 0.5, 0.2],
                               bounds=([0.0, 0.0, 0.0], [100.0, 10.0, 100.0]))
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x.copy(), branch)
    _params, _branch = best[1], best[2]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _params, _branch))