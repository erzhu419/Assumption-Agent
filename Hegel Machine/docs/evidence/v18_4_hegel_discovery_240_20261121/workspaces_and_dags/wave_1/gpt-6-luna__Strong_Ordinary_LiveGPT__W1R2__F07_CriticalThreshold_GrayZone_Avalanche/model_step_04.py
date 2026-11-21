import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        j0, beta, rho, c_sqrt = p
        return (j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
                + c_sqrt * np.sqrt(np.maximum(u, 0.0)))
    result = least_squares(lambda p: pred(p) - y, x0=np.array([5.0, 1.0, 0.7, 0.0]),
                           bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 2.0, np.inf]),
                           max_nfev=20000)
    return result.x

def predict(row):
    j0, beta, rho, c_sqrt = _fit_parameters
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
                 + c_sqrt * np.sqrt(max(u, 0.0)))

_fit_parameters = None
_original_fit = fit
def fit(train_rows):
    global _fit_parameters
    _fit_parameters = _original_fit(train_rows)
    return _fit_parameters
