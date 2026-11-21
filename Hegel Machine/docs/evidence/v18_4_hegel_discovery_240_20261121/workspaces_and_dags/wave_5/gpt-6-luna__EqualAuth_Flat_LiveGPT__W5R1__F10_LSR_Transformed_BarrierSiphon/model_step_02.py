import math
import numpy as np
from scipy.optimize import least_squares

_tau = 0.02
_params = None

def _law(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = row['u_now'] / _tau
    barrier = j0 * u_n * math.exp(-beta_bar / (0.5 + u_n))
    inhibition = 1.0 - rho * row['s_grad'] / (1.0 + row['s_grad'])
    return barrier * inhibition + c_sqrt * math.sqrt(u_n)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_law(r, p) - r['y'] for r in train_rows], dtype=float)
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_law(row, _params))