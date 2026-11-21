import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a, b, c, tau = np.exp(np.clip(p, -12, 8))
    x = float(row['x_now'])
    z = float(row['mem_conv'] if row.get('mem_conv') is not None else 0.0)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(np.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r,p) for r in rows]) - y
    starts = [np.log([0.1,0.1,1.0,0.5]), np.log([0.5,0.5,2.0,2.0]), np.log([0.02,1.0,1.0,0.1])]
    fits = [least_squares(residual,s,max_nfev=3000) for s in starts]
    return min(fits,key=lambda q: np.sum(q.fun*q.fun)).x

def predict(row):
    return float(_simulate(row, _params))