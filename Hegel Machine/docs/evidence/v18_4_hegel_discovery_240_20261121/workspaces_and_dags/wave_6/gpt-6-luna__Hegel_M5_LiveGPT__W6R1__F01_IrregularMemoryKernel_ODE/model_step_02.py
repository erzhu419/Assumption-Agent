import math
import numpy as np
from scipy.optimize import least_squares

def _hist_z(row, tau):
    pref = row.get('raw_prefix')
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        x0, x1 = xs[i-1], xs[i]
        for j in range(n):
            xa = x0 + (x1-x0) * j/n
            xb = x0 + (x1-x0) * (j+1)/n
            xm = 0.5*(xa+xb)
            # RK4 for dz/dt=(0.52*x-z)/tau with linearly interpolated x
            k1 = (0.52*xa-z)/tau
            k2 = (0.52*(0.5*(xa+xm))-(z+0.5*h*k1))/tau
            k3 = (0.52*(0.5*(xm+xb))-(z+0.5*h*k2))/tau
            k4 = (0.52*xb-(z+h*k3))/tau
            z += h*(k1+2*k2+2*k3+k4)/6
    return z

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _hist_z(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    h = H/n
    def f(x,z):
        return -a*x+b*z+c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z=f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(q):
        p = np.exp(q)
        return np.array([_forecast(r,p) for r in train_rows])-y
    starts = [np.log([0.5,0.5,0.5,0.2]), np.log([1.0,1.0,1.0,0.5]), np.log([2.0,0.2,0.8,0.1])]
    best = None
    for q0 in starts:
        res = least_squares(residual, q0, max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    return np.exp(best.x)

def predict(row):
    return float(_forecast(row, fit([row])))