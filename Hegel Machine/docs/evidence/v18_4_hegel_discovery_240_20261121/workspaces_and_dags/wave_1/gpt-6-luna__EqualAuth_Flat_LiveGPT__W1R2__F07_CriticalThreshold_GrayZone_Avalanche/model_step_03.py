import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-8, float(r['u_now'])) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # Saturating hydraulic baseline with gradient suppression, plus a
    # quadratic avalanche loss activated above a fitted gradient threshold.
    def pred(p, r):
        vmax, rho, km, threshold, amp = p
        un = max(1e-8, float(r['u_now']))
        sg = float(r['s_grad'])
        base = vmax * (1.0 - rho * sg / (1.0 + sg)) * un / (km + un)
        loss = amp * max(0.0, sg - threshold) ** 2
        return base - loss
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    result = least_squares(residual, x0=[7.0, 0.65, 1.2, 2.5, 2.0],
        bounds=([0.01, 0.0, 0.01, 0.0, 0.0], [30.0, 0.99, 20.0, 6.0, 30.0]),
        max_nfev=20000)
    predict._params = result.x
    return result.x

def predict(row):
    if not hasattr(predict, '_params'):
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, rho, km, threshold, amp = predict._params
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
    return float(base - amp * max(0.0, s - threshold) ** 2)
