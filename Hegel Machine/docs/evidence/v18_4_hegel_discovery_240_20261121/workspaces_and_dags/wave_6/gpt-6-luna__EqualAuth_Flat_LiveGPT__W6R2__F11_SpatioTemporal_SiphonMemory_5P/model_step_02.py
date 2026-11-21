import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, row):
        a_slow, b_couple, tau_fast, vmax, rho = p
        u_n = row['u_now'] / 0.005
        j = vmax * (1.0 - rho * row['s_grad'] / (1.0 + row['s_grad'])) * (u_n / (1.2 + u_n))
        x, z = float(row['x_now']), float(row['mem_conv'])
        h = float(row['H'])
        n = max(1, int(math.ceil(h / 0.001)))
        dt = h / n
        def f(x, z):
            return -a_slow*x + b_couple*z + j, (0.52*x-z)/tau_fast
        for _ in range(n):
            k1x,k1z=f(x,z)
            k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
            k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
            k4x,k4z=f(x+dt*k3x,z+dt*k3z)
            x += dt*(k1x+2*k2x+2*k3x+k4x)/6
            z += dt*(k1z+2*k2z+2*k3z+k4z)/6
        return x
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([pred(p,r) for r in rows])-y
    best=None
    for start in ([0.5,0.2,0.01,2.0,0.5],[1,0.5,0.005,3,0.3],[0.1,0.1,0.02,1,0.8]):
        res=least_squares(residual, start, bounds=([0,0,1e-5,0,-2],[10,10,1,20,2]), max_nfev=3000)
        if best is None or np.sum(res.fun**2)<np.sum(best.fun**2): best=res
    return best.x

def predict(row):
    p = fit(_MODEL_TRAIN_ROWS)
    a_slow, b_couple, tau_fast, vmax, rho = p
    u_n=row['u_now']/0.005
    j=vmax*(1-rho*row['s_grad']/(1+row['s_grad']))*(u_n/(1.2+u_n))
    x,z=float(row['x_now']),float(row['mem_conv'])
    h=float(row['H']); n=max(1,int(math.ceil(h/0.001))); dt=h/n
    def f(x,z): return -a_slow*x+b_couple*z+j,(0.52*x-z)/tau_fast
    for _ in range(n):
        k1x,k1z=f(x,z); k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2); k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2); k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x+=dt*(k1x+2*k2x+2*k3x+k4x)/6; z+=dt*(k1z+2*k2z+2*k3z+k4z)/6
    return float(x)

_MODEL_TRAIN_ROWS = []
# fit() should be called first with training rows; retained parameters support prediction.
_ORIGINAL_FIT = fit
def fit(train_rows):
    global _MODEL_TRAIN_ROWS, _MODEL_PARAMS
    _MODEL_TRAIN_ROWS = list(train_rows)
    _MODEL_PARAMS = _ORIGINAL_FIT(_MODEL_TRAIN_ROWS)
    return _MODEL_PARAMS

def predict(row):
    a_slow,b_couple,tau_fast,vmax,rho = _MODEL_PARAMS
    u_n=row['u_now']/0.005
    j=vmax*(1-rho*row['s_grad']/(1+row['s_grad']))*(u_n/(1.2+u_n))
    x,z=float(row['x_now']),float(row['mem_conv']); h=float(row['H'])
    n=max(1,int(math.ceil(h/0.001))); dt=h/n
    def f(x,z): return -a_slow*x+b_couple*z+j,(0.52*x-z)/tau_fast
    for _ in range(n):
        k1x,k1z=f(x,z); k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2); k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2); k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x+=dt*(k1x+2*k2x+2*k3x+k4x)/6; z+=dt*(k1z+2*k2z+2*k3z+k4z)/6
    return float(x)
