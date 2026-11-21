import math
import numpy as np
from scipy.optimize import least_squares

params = None

def fit(train_rows):
    global params
    def residual(p):
        j0, beta, rho, c = p
        return np.array([j0*r['u_now']*math.exp(-beta/(0.5+r['u_now']))*(1-rho*r['s_grad']/(1+r['s_grad']))+c*math.sqrt(r['u_now'])-r['y'] for r in train_rows])
    result = least_squares(residual, [3.0, 1.0, 0.5, 0.5], bounds=([0, 0, 0, 0], [np.inf, np.inf, 1, np.inf]), max_nfev=10000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    params = result.x

def predict(row):
    j0, beta, rho, c = params
    u = row['u_now']
    s = row['s_grad']
    return float(j0*u*math.exp(-beta/(0.5+u))*(1-rho*s/(1+s))+c*math.sqrt(u))