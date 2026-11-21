import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts], dtype=float)
    xs = np.array([float(p['x']) for p in pts], dtype=float)
    tau = _z0.tau
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        d = ts[i] - ts[i-1]
        # Exact exponential update for linearly interpolated x forcing.
        q = d / tau
        e = math.exp(-q)
        slope = (xs[i] - xs[i-1]) / d
        z = e*z + 0.52*(xs[i-1]*(1.0-e) + slope*(d - tau*(1.0-e)))
    return z

def _pred(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    _z0.tau = tau
    x = float(row['x_now'])
    z = _z0(row)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z = f(x,z)
        k2x,k2z = f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z = f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z = f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    def residual(p):
        return np.array([_pred(r,p)-float(r['y']) for r in train_rows])
    starts = [[0.5,0.5,0.5,0.5],[1.0,1.0,1.0,1.0],[0.2,0.2,0.8,2.0]]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.001,0.001,0.001,0.05],[5.0,5.0,5.0,10.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return float(_pred(row, predict.parameters))

predict.parameters = np.array([0.5,0.5,0.5,1.0])
_old_fit = fit
def fit(train_rows):
    predict.parameters = _old_fit(train_rows)
    return predict.parameters
