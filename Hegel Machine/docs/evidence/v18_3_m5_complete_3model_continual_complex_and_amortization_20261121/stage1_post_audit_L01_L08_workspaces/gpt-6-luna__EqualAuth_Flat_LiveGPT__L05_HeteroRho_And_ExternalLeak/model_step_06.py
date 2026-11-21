import numpy as np
from scipy.optimize import least_squares

def _sim(row, p):
    a, b, c, tau = p
    H = float(row['H']); n = max(1, int(np.ceil(H / 0.004))); dt = H / n
    x = float(row['x_now']); z = float(row['mem_conv'])
    u = float(row['u_now'])
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=f(x,z); k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2); k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2); k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    def residual(p): return np.array([_sim(r,p)-float(r['y']) for r in train_rows])
    best = None
    for start in ([0.2,0.1,0.5,0.5],[0.8,0.3,0.8,1.0],[1.5,0.5,1.0,0.2]):
        q=least_squares(residual,start,bounds=([0,0,-10,0.005],[10,10,10,10]),max_nfev=3000)
        if best is None or np.sum(q.fun**2)<np.sum(best.fun**2): best=q
    return best.x

def predict(row):
    p=fit(train_rows)
    return float(_sim(row,p))