import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.005

def _simulate(rows, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    n = len(rows)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    un = np.array([r['u_now'] / TAU_SCALE for r in rows], dtype=float)
    sg = np.array([r['s_grad'] for r in rows], dtype=float)
    drive = vmax * (1.0 - rho * sg / (1.0 + sg)) * (un / (1.2 + un))
    # Vectorized RK4 across rows; the step count resolves the fastest fitted timescale.
    steps = 256
    dt = H / steps
    for _ in range(steps):
        def deriv(xx, zz):
            return -a_slow * xx + b_couple * zz + drive, (0.52 * xx - zz) / tau_fast
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        z = z + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _simulate(train_rows, p) - y
    p0 = np.array([0.5, 0.5, 6.5, 0.70, 0.04])
    lo = np.array([0.001, -2.0, 1.0, 0.05, 0.008])
    hi = np.array([5.0, 2.0, 15.0, 0.95, 0.5])
    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=180, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_simulate([row], p)[0])