import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        j0, beta, rho, c = p
        return j0*u*np.exp(-beta/(0.5+u))*(1-rho*s/(1+s)) + c*np.sqrt(u)
    result = least_squares(lambda p: pred(p)-y, x0=[8., 1., 0.7, 0.1], bounds=([0., 0., 0., -10.], [100., 20., 1.5, 10.]))
    return result.x

def predict(row):
    p = fit([row])
    u = float(row['u_now']); s = float(row['s_grad'])
    j0, beta, rho, c = p
    return float(j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s)) + c*math.sqrt(u))