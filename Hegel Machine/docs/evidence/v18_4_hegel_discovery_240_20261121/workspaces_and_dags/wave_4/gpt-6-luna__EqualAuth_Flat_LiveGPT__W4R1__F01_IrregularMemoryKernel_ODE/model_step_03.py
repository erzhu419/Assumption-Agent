import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    pts = sorted(pts, key=lambda q: float(q['t']))
    z = 0.52 * float(pts[0]['x'])
    for p, q in zip(pts[:-1], pts[1:]):
        dt = max(0.0, float(q['t']) - float(p['t']))
        e = math.exp(-dt / tau)
        # Exponentially weighted trapezoidal approximation to the irregular input history.
        z = e * z + 0.52 * (1.0 - e) * 0.5 * (float(p['x']) + float(q['x']))
    return z

def _pred(row, pars):
    a, b, c, tau = map(float, pars)
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xx, zz):
        return -a * xx + b * zz + c * u, (0.52 * xx - zz) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return float(x)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_pred(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.5, 0.5, 0.5, 1.0]),
                           bounds=(np.array([0.001, 0.001, 0.001, 0.03]),
                                   np.array([5.0, 5.0, 5.0, 10.0])),
                           max_nfev=5000)
    _PARAMS = result.x.astype(float)
    return _PARAMS.tolist()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(row, _PARAMS))