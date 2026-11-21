import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts])
    xs = np.array([float(p['x']) for p in pts])
    T = ts[-1]
    # Exponentially weighted trapezoidal convolution, initialized at the
    # oldest observed state as specified by the kernel mechanism.
    f = 0.52 * xs * np.exp(-(T - ts) / tau) / tau
    integral = np.sum(0.5 * (f[:-1] + f[1:]) * np.diff(ts)) if len(ts) > 1 else 0.0
    return 0.52 * xs[0] * math.exp(-(T - ts[0]) / tau) + integral

def _simulate(row, pars):
    a, b, c, tau = pars
    z = _z0(row, tau)
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def rhs(xv, zv):
        return -a * xv + b * zv + c * u, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(logp):
        pars = np.exp(logp)
        return np.array([_simulate(r, pars) for r in train_rows]) - ys
    starts = [np.log([0.5, 0.5, 0.5, 0.5]), np.log([1.0, 0.2, 1.0, 1.0]), np.log([0.2, 1.0, 0.5, 1.5])]
    best = None
    for start in starts:
        res = least_squares(residual, start, max_nfev=3000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = np.exp(best.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _params))