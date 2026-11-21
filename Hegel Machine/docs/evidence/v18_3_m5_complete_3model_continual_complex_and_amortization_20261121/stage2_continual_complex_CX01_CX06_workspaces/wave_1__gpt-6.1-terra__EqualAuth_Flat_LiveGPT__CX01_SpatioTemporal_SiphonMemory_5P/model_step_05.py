import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def initial_memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    z = 0.52 * float(prefix[0]['x'])
    for p, q in zip(prefix[:-1], prefix[1:]):
        dt = float(q['t']) - float(p['t'])
        if dt <= 0:
            continue
        e = math.exp(-dt / tau)
        # Exact exponential-kernel integral for linearly interpolated x.
        x0, x1 = float(p['x']), float(q['x'])
        z = e*z + 0.52*(x0*(1-e) + (x1-x0)*(1 - tau*(1-e)/dt))
    return z

def forecast(row, p, mode):
    a, b, c, tau, vmax, branch = p
    u, s, H = float(row['u_now']), float(row['s_grad']), float(row['H'])
    source = (1.0 - 0.705*s/(1.0+s))*u/(1.2+u)
    x, z = float(row['x_now']), initial_memory(row, tau)
    n = max(1, int(math.ceil(H/0.004)))
    dt = H/n
    drive = c*u if mode == 0 else c*source
    def f(x, z):
        return -a*x + b*z + drive, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = f(x, z)
        k2, l2 = f(x+dt*k1/2, z+dt*l1/2)
        k3, l3 = f(x+dt*k2/2, z+dt*l2/2)
        k4, l4 = f(x+dt*k3, z+dt*l3)
        x += dt*(k1+2*k2+2*k3+k4)/6
        z += dt*(l1+2*l2+2*l3+l4)/6
    return x + vmax*source + branch*math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    lower = [0.0, -10.0, -10.0, 0.03, -10.0, -10.0]
    upper = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    best = None
    for mode in (0, 1):
        def residual(p):
            return np.array([forecast(r, p, mode)-float(r['y']) for r in rows])
        for tau0 in (0.2, 1.0, 3.0):
            p0 = [0.5, 0.5, 0.5, tau0, 0.5, 0.0]
            result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=1200)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, mode, result.x)
    _, _model, _params = best

def predict(row):
    return float(forecast(row, _params, _model))