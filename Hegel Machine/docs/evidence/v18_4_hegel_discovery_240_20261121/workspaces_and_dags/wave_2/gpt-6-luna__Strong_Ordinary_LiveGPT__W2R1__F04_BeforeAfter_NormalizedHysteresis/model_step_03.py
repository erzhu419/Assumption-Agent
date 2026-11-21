import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def calc(p, r):
        gain, tau, kx, ku = p
        h = max(float(r['H']), 0.0)
        return kx*float(r['x_now']) + gain*(float(r['u_now'])-float(r['u_past']))*np.exp(-h/max(tau,1e-8)) + ku*float(r['u_now'])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([calc(p, r) for r in rows]) - y
    best = None
    for seed in ([1,0.7,1,0], [-1,1,1,0], [1,2,0,0]):
        z = least_squares(residual, seed, bounds=([-20,0.02,-20,-20],[20,10,20,20]), max_nfev=10000)
        if best is None or np.sum(z.fun**2) < np.sum(best.fun**2):
            best = z
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    gain, tau, kx, ku = _PARAMS
    h = max(float(row['H']), 0.0)
    return float(kx*float(row['x_now']) + gain*(float(row['u_now'])-float(row['u_past']))*np.exp(-h/max(tau,1e-8)) + ku*float(row['u_now']))