import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    # Empirical response suggests saturating dependence on u_now, with a
    # gradient-sensitive bridge contribution. Fit a parsimonious smooth law.
    def pred(p, r):
        a,b,c,d,e = p
        u=float(r['u_now']); g=float(r['s_grad']); h=float(r['H'])
        return a + b*u/(c+u) + d*g + e*u*g
    def residual(p):
        return [pred(p,r)-float(r['y']) for r in rows]
    starts=[np.array([0.0,5.0,1.0,-0.5,0.05]),np.array([1.0,4.0,0.5,0.0,0.0])]
    fits=[least_squares(residual,s,max_nfev=5000) for s in starts]
    p=min(fits,key=lambda z: np.sum(z.fun**2)).x
    return p

def predict(row):
    # fit caches parameters on training rows; callers should invoke fit first.
    p=_params
    a,b,c,d,e=p
    u=float(row['u_now']); g=float(row['s_grad'])
    return float(a+b*u/(c+u)+d*g+e*u*g)

_params=None
_orig_fit=fit
def fit(train_rows):
    global _params
    _params=_orig_fit(train_rows)
    return _params
