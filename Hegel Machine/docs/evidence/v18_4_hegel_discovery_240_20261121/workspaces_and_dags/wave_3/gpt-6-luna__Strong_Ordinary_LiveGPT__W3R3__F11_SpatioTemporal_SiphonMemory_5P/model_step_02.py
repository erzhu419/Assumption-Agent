import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    a, b, c, tau, vmax, rho = p
    # Integrate coupled states x,z with fixed step RK4.
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    def f(x, z):
        return (-a*x + b*z + J, (0.52*x-z)/tau)
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    def residual(p):
        return np.array([_pred(r,p)-float(r['y']) for r in train_rows])
    starts = [[.3,.2,.3,.5,2.,.7],[.6,.1,.5,1.,3.,.5],[.1,.4,.2,.2,1.,.9]]
    best=None
    for x0 in starts:
        sol=least_squares(residual,x0,bounds=([0,0,-10,.01,0,0],[5,5,10,10,20,2]),max_nfev=3000)
        if best is None or np.sum(sol.fun**2)<np.sum(best.fun**2): best=sol
    return best.x

def predict(row):
    return float(_pred(row, fit([dict(row, y=0.0)])))