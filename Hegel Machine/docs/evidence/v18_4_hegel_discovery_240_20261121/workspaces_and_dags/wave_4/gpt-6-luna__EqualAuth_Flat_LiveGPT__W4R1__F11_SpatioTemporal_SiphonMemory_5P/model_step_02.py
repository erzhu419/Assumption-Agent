import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now'])  # tau_scale = 1.0
    sg = float(row['s_grad'])
    j = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    n = max(40, int(math.ceil(240 * H)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + j, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.8, 0.3, 2.0, 0.4, 0.7]),
        bounds=(np.array([0.001, 0.0, 0.001, 0.0, 0.02]),
                np.array([5.0, 5.0, 10.0, 1.0, 10.0])),
        max_nfev=5000)
    _PARAMS = result.x
    return {'a_slow': float(_PARAMS[0]), 'b_couple': float(_PARAMS[1]),
            'vmax': float(_PARAMS[2]), 'rho': float(_PARAMS[3]),
            'tau_fast': float(_PARAMS[4])}

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _PARAMS))