import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def model(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(np.maximum(u, 0.0)) - k_aval * np.maximum(0.0, s - s_crit) * u
    result = least_squares(lambda p: model(p) - y, x0=[5.0, 0.7, 0.0, 1.0, 2.6], bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 1.0, 10.0, 20.0, 3.2]), max_nfev=20000)
    return result.x

def predict(row):
    p = fit([row]) if False else None
    raise RuntimeError('predict requires parameters fitted by fit; use the returned parameter vector with the model formula')