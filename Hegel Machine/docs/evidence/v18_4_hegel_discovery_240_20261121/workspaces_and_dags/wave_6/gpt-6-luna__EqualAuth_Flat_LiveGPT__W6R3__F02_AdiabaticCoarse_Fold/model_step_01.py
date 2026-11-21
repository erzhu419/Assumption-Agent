import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Coarse relaxation toward x_ss = k_ss * u_now; transient decays at rate a_eff.
    U = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    X = np.asarray([float(r['x_now']) for r in train_rows], dtype=float)
    H = np.asarray([float(r['H']) for r in train_rows], dtype=float)
    Y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        k, a = p
        return k*U + (X-k*U)*np.exp(-a*H) - Y
    result = least_squares(residual, x0=np.array([2.4, 1.0]), bounds=([-100.0, 0.0], [100.0, 100.0]))
    return {'k_ss': float(result.x[0]), 'a_eff': float(result.x[1])}

def predict(row):
    k = float(_params['k_ss'])
    a = float(_params['a_eff'])
    u = float(row['u_now'])
    x = float(row['x_now'])
    h = float(row['H'])
    return float(k*u + (x-k*u)*math.exp(-a*h))