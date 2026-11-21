import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z_from_prefix(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        t0, t1 = float(left['t']), float(right['t'])
        x0, x1 = float(left['x']), float(right['x'])
        dt = t1 - t0
        if dt <= 0:
            continue
        # RK4 for tau*dz/dt = 0.52*x(t)-z with x linearly interpolated.
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for j in range(n):
            q = j / n
            qh = (j + 0.5) / n
            q1 = (j + 1.0) / n
            xa = x0 + (x1 - x0) * q
            xm = x0 + (x1 - x0) * qh
            xb = x0 + (x1 - x0) * q1
            f1 = (0.52 * xa - z) / tau
            f2 = (0.52 * xm - (z + 0.5*h*f1)) / tau
            f3 = (0.52 * xm - (z + 0.5*h*f2)) / tau
            f4 = (0.52 * xb - (z + h*f3)) / tau
            z += h * (f1 + 2*f2 + 2*f3 + f4) / 6.0
    return z

def _forecast(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _z_from_prefix(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def deriv(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z = deriv(x,z)
        k2x,k2z = deriv(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z = deriv(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z = deriv(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_forecast(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.5, 0.5, 0.5, 1.0]),
                           bounds=([1e-5, 1e-5, 1e-5, 0.03], [5.0, 5.0, 5.0, 10.0]),
                           max_nfev=3000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_forecast(row, _PARAMS))