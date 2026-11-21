import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(np.ceil(H / 0.004)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a*xx + b*zz + c*u, (0.52*xx - zz)/tau])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5*dt*k1)
        k3 = f(state + 0.5*dt*k2)
        k4 = f(state + dt*k3)
        state += (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    starts = [np.array([0.8, 0.5, 1.0, 0.8]), np.array([1.5, 1.0, 0.7, 1.5]), np.array([0.3, 1.2, 1.2, 0.4])]
    fits = [least_squares(residual, s, bounds=([0.001, 0.001, 0.001, 0.03], [8.0, 8.0, 8.0, 10.0]), max_nfev=3000) for s in starts]
    return min(fits, key=lambda q: np.sum(q.fun**2)).x

def predict(row):
    return _simulate(row, _params)
