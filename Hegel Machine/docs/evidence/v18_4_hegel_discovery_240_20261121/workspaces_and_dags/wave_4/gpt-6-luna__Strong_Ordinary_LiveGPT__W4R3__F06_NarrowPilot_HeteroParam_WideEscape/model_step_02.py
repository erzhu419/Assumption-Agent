import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    un = u  # tau_scale = 1.0
    def cfun(p, u, s):
        vmax, rho, c = p[:3]
        return vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u)) + c*np.sqrt(np.maximum(u, 0.0))
    def qfun(p, u, s):
        vmax, kh, alpha, leak = p
        return vmax*u*u/(kh+u*u)/(1.0+alpha*s)-leak*u
    def rfun(p, u, s):
        j0, beta, rho, c = p
        return j0*u*np.exp(-beta/(0.5+u))*(1.0-rho*s/(1.0+s))+c*np.sqrt(np.maximum(u,0.0))
    candidates = []
    for fn, starts, lo, hi, name in [
        (cfun, [[8,0.5,1],[10,0.7,1],[5,0.2,2]], [0,0,-20], [100,2,20], 'C'),
        (qfun, [[10,1,0.5,0.1],[20,2,0.2,0.1],[8,0.5,1,0]], [0,1e-5,0,0], [200,100,10,20], 'Q'),
        (rfun, [[10,0.5,0.5,1],[20,1,0.3,1],[8,0.1,0.8,2]], [0,0,0,-20], [200,20,2,20], 'R')]:
        best = None
        for p0 in starts:
            res = least_squares(lambda p: fn(p, un, s)-y, p0, bounds=(lo,hi), max_nfev=5000)
            score = float(np.dot(res.fun,res.fun))
            if best is None or score < best[0]: best = (score, res.x)
        candidates.append((best[0], name, best[1]))
    _, name, params = min(candidates, key=lambda z:z[0])
    return {'family': name, 'params': params.tolist()}

def predict(row):
    p = np.asarray(row['_model_params'], dtype=float)
    u = float(row['u_now']); s = float(row['s_grad'])
    family = row['_model_family']
    if family == 'C':
        return float(p[0]*(1-p[1]*s/(1+s))*u/(1.2+u)+p[2]*math.sqrt(max(u,0.0)))
    if family == 'Q':
        return float(p[0]*u*u/(p[1]+u*u)/(1+p[2]*s)-p[3]*u)
    return float(p[0]*u*math.exp(-p[1]/(0.5+u))*(1-p[2]*s/(1+s))+p[3]*math.sqrt(max(u,0.0)))