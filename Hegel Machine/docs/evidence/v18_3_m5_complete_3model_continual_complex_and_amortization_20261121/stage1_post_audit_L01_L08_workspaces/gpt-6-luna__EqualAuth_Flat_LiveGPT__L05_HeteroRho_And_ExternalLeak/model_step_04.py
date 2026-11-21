import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho0, rho_x, rho_mem, leak = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    sat = u / (1.2 + u)
    rho = rho0 + rho_x * float(row['x_now']) + rho_mem * float(row['mem_conv'])
    basin = vmax * (1.0 - rho * s / (1.0 + s)) * sat
    return basin - leak * max(0.0, float(row['H']) - 0.6)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 0.7, 0.0, 0.0, 0.0]),
                           bounds=([0.01, 0.0, -1.0, -1.0, -0.1],
                                   [10.0, 1.5, 1.0, 1.0, 0.1]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))