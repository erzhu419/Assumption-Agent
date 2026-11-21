import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    field = u / (1.2 + u)
    return (vmax * sink * field * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    # Broad positive timescale bounds with fast < slow; fit all five parameters.
    result = least_squares(residual, x0=np.array([6.0, 0.70, 0.7, 0.35, 12.0]),
                           bounds=(np.array([0.01, 0.0, 0.0, 0.01, 1.0]),
                                   np.array([30.0, 0.999, 10.0, 5.0, 200.0])),
                           max_nfev=50000)
    _params = result.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))