import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Four barrier-flux parameters plus a fifth transient-lag gain.
    def pred(p, r):
        j0, beta, rho, c_sqrt, lag = p
        u = max(1e-4, float(r['u_now']))
        s = max(0.0, float(r['s_grad']))
        base = j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c_sqrt*math.sqrt(u)
        # Memory convolution and the changing field encode the observed lag.
        lag_signal = (float(r['mem_conv']) - float(r['x_now'])) * float(r['H'])
        return base * math.exp(lag * lag_signal)
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 0.5, 0.5, 0.3, 0.1]), bounds=([0.0, 0.0, 0.0, 0.0, -5.0], [20.0, 10.0, 1.0, 10.0, 5.0]), max_nfev=10000)
    return result.x.tolist()

def predict(row):
    p = fit([dict(row, y=row['y'])]) if 'y' in row else None
    if p is None:
        raise ValueError('predict requires parameters fitted through fit')
    j0, beta, rho, c_sqrt, lag = p
    u = max(1e-4, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    base = j0*u*math.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s)) + c_sqrt*math.sqrt(u)
    return float(base * math.exp(lag*(float(row['mem_conv'])-float(row['x_now']))*float(row['H'])))