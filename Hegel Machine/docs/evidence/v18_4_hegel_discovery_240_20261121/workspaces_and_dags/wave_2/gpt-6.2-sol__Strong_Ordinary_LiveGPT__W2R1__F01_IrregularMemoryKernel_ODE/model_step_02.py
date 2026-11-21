import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _prepare(row):
    prefix = row.get('raw_prefix')
    z = row.get('mem_conv')
    history = None
    if z is None and prefix:
        points = sorted(prefix, key=lambda point: point['t'])
        t = np.array([point['t'] for point in points], dtype=float)
        x = np.array([point['x'] for point in points], dtype=float)
        history = (t[-1] - t, np.diff(t), x)
    return float(row['x_now']), float(row['u_now']), float(row['H']), z, history

def _forecast(p, data):
    a, b, c, tau = map(float, p)
    x, u, H, z, history = data
    if z is None:
        if history is None:
            z = 0.52 * x
        else:
            ages, widths, samples = history
            w = np.exp(-ages / tau) * samples
            z = 0.52 * float(w[0] + np.dot(widths, w[:-1] + w[1:]) / (2.0 * tau))
    z = float(z)
    drive = c * u
    n = max(1, math.ceil(H / 0.004))
    h = H / n
    def rhs(x, z):
        return -a * x + b * z + drive, (0.52 * x - z) / tau
    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + h*k1/2, z + h*l1/2)
        k3, l3 = rhs(x + h*k2/2, z + h*l2/2)
        k4, l4 = rhs(x + h*k3, z + h*l3)
        x += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        z += h * (l1 + 2*l2 + 2*l3 + l4) / 6
    return float(x)

def fit(train_rows):
    global _params
    rows = [(_prepare(row), float(row['y'])) for row in train_rows]
    def residuals(p):
        return np.array([_forecast(p, data) - y for data, y in rows])
    result = least_squares(residuals, [2.1, 1.2, 1.2, 0.75], bounds=([0.01, 0.0, 0.0, 0.02], [20.0, 20.0, 20.0, 10.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    return _forecast(_params, _prepare(row))
