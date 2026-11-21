import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row):
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    # Integrate the archived exponential memory kernel over the observed history.
    z = 0.52 * float(pref[0]['x'])
    t_prev = float(pref[0]['t'])
    x_prev = float(pref[0]['x'])
    for p in pref[1:]:
        t = float(p['t'])
        x = float(p['x'])
        dt = max(0.0, t - t_prev)
        if dt:
            e = math.exp(-dt / 1.0)
            z = e * z + 0.52 * (0.5 * (x_prev + x)) * (1.0 - e)
        t_prev, x_prev = t, x
    return z

def _simulate(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row)
    u = float(row['u_now'])
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xx, zz):
        return (-a * xx + b * zz + c * u, (0.52 * xx - zz) / tau)
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def unpack(q):
        return (math.exp(float(q[0])), float(q[1]), float(q[2]), math.exp(float(q[3])))
    def residual(q):
        p = unpack(q)
        return np.asarray([_simulate(r, p) for r in rows]) - y
    result = least_squares(residual, np.array([math.log(0.5), 0.2, 0.5, math.log(0.2)]), max_nfev=5000)
    return unpack(result.x)

def predict(row):
    return float(_simulate(row, _PARAMS))