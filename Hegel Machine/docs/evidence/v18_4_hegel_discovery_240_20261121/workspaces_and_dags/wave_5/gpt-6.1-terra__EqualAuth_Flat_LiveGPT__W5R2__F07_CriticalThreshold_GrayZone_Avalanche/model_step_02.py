import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(max(0.0, u)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    def residuals(p):
        return np.array([_evaluate(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residuals, [5.0, 0.3, 0.5, 2.0, 2.6], bounds=([0.0, 0.0, -10.0, 0.0, 2.44], [50.0, 5.0, 10.0, 50.0, 2.78]), max_nfev=20000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_evaluate(row, _params))