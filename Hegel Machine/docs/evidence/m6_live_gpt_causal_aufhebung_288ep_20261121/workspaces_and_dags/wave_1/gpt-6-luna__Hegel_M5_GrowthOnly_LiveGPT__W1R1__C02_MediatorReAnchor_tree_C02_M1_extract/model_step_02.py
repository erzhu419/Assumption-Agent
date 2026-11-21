import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Fit the measured thermal deployment law using a saturating surface term,
    # raw-input square-root contribution, and thermal modulation of that term.
    X = np.array([[r['s_surf'], r['u_raw'], r['t_proc'], r['m_1'], r['m_2'], r['m_3']] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, row):
        s,u,t,m1,m2,m3 = row
        vmax,k,c,b1,b2,b3,dt = p
        return vmax*s/(k+s) + c*math.sqrt(max(u,0.0)) + b1*m1 + b2*m2 + b3*m3 + dt*t
    def residual(p):
        return np.array([pred(p, row)-yy for row,yy in zip(X,y)])
    starts = [np.array([7.,1.,0.,0.,0.,0.,0.]), np.zeros(7)+0.1]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, max_nfev=20000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    # Deterministic fit-on-first-call convention; callers may set parameters via fit.
    global _params
    if '_params' not in globals():
        _params = None
    if _params is None:
        return float('nan')
    vmax,k,c,b1,b2,b3,dt = _params
    s,u,t = row['s_surf'],row['u_raw'],row['t_proc']
    return float(vmax*s/(k+s) + c*math.sqrt(max(u,0.0)) + b1*row['m_1'] + b2*row['m_2'] + b3*row['m_3'] + dt*t)
