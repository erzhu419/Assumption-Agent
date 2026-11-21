import math
import numpy as np
from scipy.optimize import least_squares

_MODEL_PARAMS = None

def _initial_z(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    hist = row.get('raw_prefix') or []
    if len(hist) < 2:
        return 0.52 * float(row['x_now'])
    ts = np.asarray([float(p['t']) for p in hist], dtype=float)
    xs = np.asarray([float(p['x']) for p in hist], dtype=float)
    z = 0.52 * xs[0]
    # Integrate tau dz/dt = 0.52*x(t)-z with RK4 over the irregular history.
    for j in range(len(ts) - 1):
        dt = ts[j + 1] - ts[j]
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for k in range(n):
            xa = xs[j] + (xs[j + 1] - xs[j]) * (k / n)
            xb = xs[j] + (xs[j + 1] - xs[j]) * ((k + 1) / n)
            xm = 0.5 * (xa + xb)
            f = lambda zz, xx: (0.52 * xx - zz) / tau
            k1 = f(z, xa)
            k2 = f(z + 0.5*h*k1, xm)
            k3 = f(z + 0.5*h*k2, xm)
            k4 = f(z + h*k3, xb)
            z += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return float(z)

def _simulate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = _initial_z(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def deriv(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + .5*h*k1x, z + .5*h*k1z)
        k3x, k3z = deriv(x + .5*h*k2x, z + .5*h*k2z)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x += h*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += h*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return float(x)

def fit(train_rows):
    global _MODEL_PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([0.7, 0.3, 0.5, 0.4]), bounds=([0.001, 0.001, 0.001, 0.01], [5.0, 5.0, 5.0, 5.0]), max_nfev=3000)
    _MODEL_PARAMS = result.x.copy()
    return _MODEL_PARAMS

def predict(row):
    if _MODEL_PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _MODEL_PARAMS))