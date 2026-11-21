import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    tau = 0.05
    def calc(p, r):
        vmax, rho, c_loc = p
        u = r['u_now'] / tau
        s = r['s_grad']
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return base + c_loc * math.sqrt(max(u, 0.0))
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([calc(p, r) for r in rows]) - y
    starts = [[6.5, 0.70, 0.0], [5.0, 0.70, 0.5], [8.0, 0.70, -0.5]]
    fits = [least_squares(residual, x0, bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])) for x0 in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    return {'params': best.x.tolist(), 'tau': tau}

def predict(row):
    vmax, rho, c_loc = _model['params']
    u = row['u_now'] / _model['tau']
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)))

_model = {'params': [6.5, 0.70, 0.0], 'tau': 0.05}
# fit() returns parameters; bind them for predict() after training.
def fit_and_bind(train_rows):
    global _model
    _model = fit(train_rows)
    return _model
fit = fit_and_bind
