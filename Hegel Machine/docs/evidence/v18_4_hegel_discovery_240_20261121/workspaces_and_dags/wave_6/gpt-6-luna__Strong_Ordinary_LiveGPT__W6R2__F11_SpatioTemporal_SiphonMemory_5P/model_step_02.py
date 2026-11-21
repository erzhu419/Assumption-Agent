import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.005

def _rk4(row, p):
    a_slow, b_couple, c_drive, tau_fast, rho, vmax = p
    # Use the archived spatio-temporal siphon-memory law; fit the two
    # siphon parameters as well as the three ODE coefficients and tau_fast.
    u_n = float(row['u_now']) / TAU_SCALE
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return (-a_slow * xv + b_couple * zv + c_drive * float(row['u_now']) + j,
                (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    # The archive labels this family 5-param; rho and vmax are nuisance
    # combinations with the stated three ODE parameters and fast time scale.
    # Fit all six effective parameters to avoid assuming archived source values.
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_rk4(r, p) for r in train_rows]) - y
    starts = [np.array([0.5, 0.2, 10.0, 0.03, 0.5, 2.0]),
              np.array([1.0, 0.1, 20.0, 0.08, 0.7, 3.0]),
              np.array([0.2, 0.4, 5.0, 0.015, 0.3, 1.5])]
    lo = np.array([0.0, -5.0, -100.0, 0.001, 0.0, 0.0])
    hi = np.array([10.0, 5.0, 100.0, 2.0, 1.5, 30.0])
    results = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=3000) for x0 in starts]
    best = min(results, key=lambda r: float(np.dot(r.fun, r.fun)))
    return best.x.tolist()

def predict(row):
    return float(_rk4(row, _FIT_PARAMS))

_FIT_PARAMS = [0.5, 0.2, 10.0, 0.03, 0.5, 2.0]
# Bind fitted parameters when fit() is called, while preserving predict(row).
_fit_impl = fit
def fit(train_rows):
    global _FIT_PARAMS
    _FIT_PARAMS = _fit_impl(train_rows)
    return _FIT_PARAMS
