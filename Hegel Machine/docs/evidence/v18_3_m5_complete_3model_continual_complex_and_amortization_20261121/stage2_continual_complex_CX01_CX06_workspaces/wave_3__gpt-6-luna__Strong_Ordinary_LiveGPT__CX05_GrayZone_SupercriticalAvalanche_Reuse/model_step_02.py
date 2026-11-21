import math
import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # characteristic_tau_scale = 1.0
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    best = None
    # Threshold starts span the documented supercritical range; select by data fit.
    for threshold in (1.30, 1.40, 1.50):
        result = least_squares(
            residual, x0=np.array([5.0, 0.7, 0.2, 1.0, threshold]),
            bounds=(np.array([0.0, 0.0, -10.0, 0.0, 1.0]),
                    np.array([100.0, 1.0, 10.0, 100.0, 2.0])),
            max_nfev=20000
        )
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return {'params': best[1]}

def predict(row):
    return float(_predict_params(row, fit._model['params']))

# fit stores the fitted parameters for subsequent predict calls.
_fit_impl = fit
def fit(train_rows):
    model = _fit_impl(train_rows)
    fit._model = model
    return model
