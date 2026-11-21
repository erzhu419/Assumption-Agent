import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    prefix = row.get('raw_prefix') or []
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    if len(prefix) < 2:
        return 0.52 * float(row['x_now'])
    ts = np.array([p['t'] for p in prefix], dtype=float)
    xs = np.array([p['x'] for p in prefix], dtype=float)
    z = 0.52 * xs[0]
    end = ts[-1]
    for i in range(len(ts)-1):
        dt = ts[i+1] - ts[i]
        if dt <= 0:
            continue
        # Exact exponential-kernel integral for linearly interpolated x(t).
        age = end - ts[i+1]
        decay = math.exp(-dt / tau)
        slope = (xs[i+1] - xs[i]) / dt
        forcing = xs[i] * (1.0-decay) + slope * (dt - tau*(1.0-decay))
        z = z * decay + 0.52 * math.exp(-age/tau) * forcing
    return z

def _simulate(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a*xx + b*zz + c*u, (0.52*xx-zz)/tau])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1=f(state); k2=f(state+0.5*dt*k1); k3=f(state+0.5*dt*k2); k4=f(state+dt*k3)
        state += (dt/6.0)*(k1+2*k2+2*k3+k4)
    return float(state[0])

def fit(train_rows):
    def residual(p):
        return np.array([_simulate(r,p)-float(r['y']) for r in train_rows])
    starts = [[0.8,0.4,1.0,0.15],[1.0,0.2,0.8,0.3],[0.5,0.8,1.2,0.08]]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.001,0.0,0.0,0.005],[10.0,10.0,10.0,2.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return _simulate(row, fit([row]))