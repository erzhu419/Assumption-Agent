import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a, b, vmax, rho, tau = p
    x, z = float(row['x_now']), float(row['mem_conv'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    n = max(20, int(np.ceil(H / 0.005)))
    dt = H / n
    def deriv(x, z):
        return -a*x + b*z + drive, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=deriv(x,z)
        k2x,k2z=deriv(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=deriv(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=deriv(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    rows=list(train_rows)
    y=np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_simulate(r,p) for r in rows])-y
    starts=[[0.5,0.5,6.5,0.7,0.5],[1.0,1.0,5.0,0.5,1.0],[0.2,-0.5,8.0,0.8,0.25]]
    fits=[least_squares(residual,x0,bounds=([0.001,-3,0.01,0.001,0.01],[5,3,30,0.999,5]),max_nfev=3000) for x0 in starts]
    return min(fits,key=lambda f: np.sum(f.fun**2)).x

def predict(row):
    return float(_simulate(row, _fitted_params))
