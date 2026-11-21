import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-8, r['u_now']) for r in train_rows], dtype=float)
    s = np.array([max(0.0, r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    # Saturating u response modulated by sink gradient; all parameters are fit numerically.
    def pred(p, uu, ss):
        a, b, c, k, d = p
        base = a + b*np.sqrt(uu) + c*uu/(k+uu)
        return base*np.exp(-d*ss)
    p0 = np.array([1.0, 1.0, 1.0, 0.08, 0.2])
    lo = np.array([-20.0, -20.0, -20.0, 1e-5, -5.0])
    hi = np.array([20.0, 20.0, 20.0, 5.0, 5.0])
    res = least_squares(lambda p: pred(p,u,s)-y, p0, bounds=(lo,hi), max_nfev=5000)
    return res.x.tolist()

def predict(row):
    p = fit(_rows_for_fit)
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    a,b,c,k,d = p
    return float((a+b*math.sqrt(u)+c*u/(k+u))*math.exp(-d*s))