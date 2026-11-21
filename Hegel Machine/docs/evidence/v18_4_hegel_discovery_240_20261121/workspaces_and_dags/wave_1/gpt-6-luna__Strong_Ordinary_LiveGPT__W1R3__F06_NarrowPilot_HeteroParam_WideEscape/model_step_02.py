import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    best = None
    # Multiple numerical starts reduce sensitivity to the nonlinear parameterization.
    for start in ([2.0, 0.5, 0.7, 0.0], [4.0, 1.5, 0.5, -0.5], [1.0, 0.1, 0.9, 1.0]):
        res = least_squares(lambda p: np.asarray([_model(r, p) for r in rows]) - y,
                            x0=np.asarray(start, dtype=float),
                            bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 20.0, 5.0, 20.0]),
                            max_nfev=20000)
        loss = float(np.sum(res.fun ** 2))
        if best is None or loss < best[0]:
            best = (loss, res.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_model(row, _PARAMS))