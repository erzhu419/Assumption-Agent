import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # tau_scale = 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    return base + c_loc * np.sqrt(max(u_n, 0.0)) - k_aval * max(0.0, s - s_crit) * u_n

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    result = least_squares(
        residual,
        x0=np.array([6.0, 0.7, 0.0, 2.0, 2.6]),
        bounds=(np.array([0.0, 0.0, -np.inf, 0.0, 2.0]),
                np.array([np.inf, 1.0, np.inf, np.inf, 3.2])),
        max_nfev=20000
    )
    return {'params': result.x.tolist()}

def predict(row):
    return float(_predict_params(row, fit_params))