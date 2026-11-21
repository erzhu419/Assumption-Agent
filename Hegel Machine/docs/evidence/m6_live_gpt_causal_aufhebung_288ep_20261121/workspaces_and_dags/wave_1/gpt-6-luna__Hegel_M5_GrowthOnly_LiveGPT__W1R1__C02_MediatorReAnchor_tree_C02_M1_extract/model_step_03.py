import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['s_surf']), math.sqrt(max(0.0, float(r['u_raw']))), float(r['t_proc']), float(r['m_1']), float(r['m_2']), float(r['m_3'])] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    # Bounded saturating surface term + additive raw-input, thermal, and mediator effects.
    def pred(p, x):
        vmax, km, c_raw, c_t, c1, c2, c3 = p
        s, root_u, t, m1, m2, m3 = x
        return vmax * s / (km + s) + c_raw * root_u + c_t * t + c1*m1 + c2*m2 + c3*m3
    def resid(p):
        return np.array([pred(p, x) for x in X]) - y
    p0 = np.array([5.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    lo = np.array([0.0, 1e-8, -20.0, -20.0, -20.0, -20.0, -20.0])
    hi = np.array([30.0, 30.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    opt = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)
    # Save coefficients on the function object for the required predict API.
    fit.params = opt.x
    return fit

def predict(row):
    p = fit.params
    s = float(row['s_surf']); u = max(0.0, float(row['u_raw']))
    t = float(row['t_proc'])
    return float(p[0]*s/(p[1]+s) + p[2]*math.sqrt(u) + p[3]*t + p[4]*float(row['m_1']) + p[5]*float(row['m_2']) + p[6]*float(row['m_3']))
