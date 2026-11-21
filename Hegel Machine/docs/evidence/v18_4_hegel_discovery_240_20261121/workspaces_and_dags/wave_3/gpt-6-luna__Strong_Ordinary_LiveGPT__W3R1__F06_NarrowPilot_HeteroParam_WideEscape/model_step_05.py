import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_branch = None

def fit(train_rows):
    global _params, _branch
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    candidates = []
    # Compare documented open square-root branch with the mass-drift leak branch.
    for branch in ('sqrt', 'leak'):
        def pred(p):
            vmax, rho, c = p
            base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            return base + c * np.sqrt(u) if branch == 'sqrt' else base - c * u
        p0 = np.array([6.0, 0.7, 0.5])
        result = least_squares(lambda p: pred(p) - y, p0,
                               bounds=([0.0, 0.0, -10.0 if branch == 'sqrt' else 0.0],
                                       [100.0, 2.0, 10.0]))
        candidates.append((float(np.mean((pred(result.x) - y) ** 2)), branch, result.x))
    _, _branch, _params = min(candidates, key=lambda item: item[0])
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    vmax, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return float(base + c * math.sqrt(u) if _branch == 'sqrt' else base - c * u)
