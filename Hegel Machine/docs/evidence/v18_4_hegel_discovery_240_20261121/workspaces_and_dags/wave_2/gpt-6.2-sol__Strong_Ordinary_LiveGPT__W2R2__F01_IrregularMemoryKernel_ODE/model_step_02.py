import math
import numpy as np
from scipy.optimize import least_squares


def _prepare(row):
    prefix = row.get('raw_prefix')
    if prefix:
        xs = tuple(float(p['x']) for p in prefix)
        ts = tuple(float(p['t']) for p in prefix)
        intervals = tuple((t1-t0, x0, x1-x0) for t0, t1, x0, x1 in zip(ts[:-1], ts[1:], xs[:-1], xs[1:]) if t1 > t0)
        first_x = xs[0]
    else:
        first_x, intervals = float(row['x_now']), ()
    return float(row['x_now']), float(row['u_now']), float(row['H']), row.get('mem_conv'), first_x, intervals


def _forecast(params, data):
    a, b, c, tau = map(float, params)
    x, u, H, memory, first_x, intervals = data
    if memory is not None:
        z = float(memory)
    else:
        z = 0.52 * first_x
        # Exact exponential convolution of linearly interpolated history.
        for dt, x0, dx in intervals:
            q = dt / tau
            w = -math.expm1(-q)
            ramp = q * (0.5 - q / 6.0) if q < 1e-5 else 1.0 - w / q
            z += w * (0.52 * x0 - z) + 0.52 * ramp * dx
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    drive, inv_tau = c * u, 1.0 / tau

    def rhs(x, z):
        return -a*x + b*z + drive, (0.52*x-z)*inv_tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5*h*k1, z + 0.5*h*l1)
        k3, l3 = rhs(x + 0.5*h*k2, z + 0.5*h*l2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        z += h*(l1 + 2*l2 + 2*l3 + l4)/6.0
    return float(x)


def fit(train_rows):
    global _params
    rows = [(_prepare(row), float(row['y'])) for row in train_rows]

    def residuals(params):
        return np.array([_forecast(params, data) - y for data, y in rows])

    result = least_squares(residuals, [1.0, 0.8, 1.0, 0.5], bounds=([0.0, 0.0, 0.0, 0.01], [np.inf, np.inf, np.inf, np.inf]), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params


def predict(row) -> float:
    return _forecast(_params, _prepare(row))
