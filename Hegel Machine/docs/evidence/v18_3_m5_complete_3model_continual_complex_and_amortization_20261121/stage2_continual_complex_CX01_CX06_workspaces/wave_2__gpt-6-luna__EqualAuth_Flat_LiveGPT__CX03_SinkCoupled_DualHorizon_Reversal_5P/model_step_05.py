import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def simulate(row, p):
        vmax, rho, a, b, tau = p
        u = float(row['u_now'])
        s = float(row['s_grad'])
        flux = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        x = float(row['x_now'])
        z = float(row['mem_conv'] or 0.0)
        H = float(row['H'])
        n = max(1, int(math.ceil(H / 0.08)))
        dt = H / n
        def f(q):
            xx, zz = q
            return np.array([-a * xx + b * zz + flux, (0.52 * xx - zz) / tau])
        q = np.array([x, z], dtype=float)
        for _ in range(n):
            k1 = f(q)
            k2 = f(q + 0.5 * dt * k1)
            k3 = f(q + 0.5 * dt * k2)
            k4 = f(q + dt * k3)
            q += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return q[0]
    def residual(p):
        return np.array([simulate(r, p) - float(r['y']) for r in rows])
    best = None
    starts = [np.array(v, dtype=float) for v in (
        (1.0, 0.7, 0.12, 0.08, 1.0),
        (2.0, 0.5, 0.2, 0.15, 2.0),
        (3.0, 0.8, 0.05, 0.03, 0.5))]
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.001, 0.0, 0.001, 0.0, 0.03], [20.0, 1.5, 3.0, 3.0, 30.0]), max_nfev=180)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    p = fit(_train_rows)
    vmax, rho, a, b, tau = p
    u = float(row['u_now']); s = float(row['s_grad'])
    flux = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    q = np.array([float(row['x_now']), float(row['mem_conv'] or 0.0)], dtype=float)
    H = float(row['H']); n = max(1, int(math.ceil(H / 0.08))); dt = H / n
    def f(v):
        return np.array([-a*v[0] + b*v[1] + flux, (0.52*v[0]-v[1])/tau])
    for _ in range(n):
        k1=f(q); k2=f(q+0.5*dt*k1); k3=f(q+0.5*dt*k2); k4=f(q+dt*k3)
        q += (dt/6.0)*(k1+2*k2+2*k3+k4)
    return float(q[0])