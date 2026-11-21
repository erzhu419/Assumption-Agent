import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.005

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    un = float(row['u_now']) / TAU_SCALE
    sg = float(row['s_grad'])
    j = vmax * (1.0 - rho * sg / (1.0 + sg)) * (un / (1.2 + un))
    n = max(1, int(np.ceil(H / max(tau_fast, 1e-5) * 16.0)))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + j, (0.52 * xv - zv) / tau_fast
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
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    starts = [np.array([0.5, 0.5, 2.0, 0.4, 0.03]), np.array([1.0, 0.2, 3.0, 0.6, 0.01])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, 0.0, 0.001], [10.0, 10.0, 20.0, 1.5, 0.2]), max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_simulate(row, _MODEL_PARAMS))

_MODEL_PARAMS = np.array([0.5, 0.5, 2.0, 0.4, 0.03])
# fit(train_rows) returns parameters; install them before using predict.
def fit_and_install(train_rows):
    global _MODEL_PARAMS
    _MODEL_PARAMS = fit(train_rows)
    return _MODEL_PARAMS
