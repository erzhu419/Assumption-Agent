import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    vmax, km, rho, power = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    sp = s ** power
    return vmax * u / (km + u) * (1.0 - rho * sp / (1.0 + sp))

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(p, r) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [8.0, 1.2, 0.7, 1.5], bounds=([0.0, 0.001, 0.0, 0.1], [100.0, 100.0, 0.999, 10.0]), max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    return float(_model(_params, row))
