import math
import numpy as np
from scipy.optimize import least_squares

H_TARGET = 0.6
TAU_GAUGE = 0.25
KEYS = ('x_now', 'mem_conv', 'u_now')
_coeff = None
_params = None

def terminal_map(params, H):
    a, b, c = params
    state = np.array([[1., 0., 0.], [0., 1., 0.]])
    drive = np.array([0., 0., c])
    def rhs(q):
        x, z = q
        return np.array([-a*x + b*z + drive, (0.52*x - z)/TAU_GAUGE])
    n = max(1, math.ceil(H/0.004))
    dt = H/n
    for _ in range(n):
        k1 = rhs(state)
        k2 = rhs(state + 0.5*dt*k1)
        k3 = rhs(state + 0.5*dt*k2)
        k4 = rhs(state + dt*k3)
        state += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return state[0]

def fit(train_rows):
    global _params, _coeff
    X = np.array([[row[k] for k in KEYS] for row in train_rows])
    y = np.array([row['y'] for row in train_rows])
    hs = np.array([row['H'] for row in train_rows])
    groups = [(float(h), hs == h) for h in np.unique(hs)]
    def residual(params):
        return np.concatenate([X[mask] @ terminal_map(params, h) - y[mask] for h, mask in groups])
    _params = least_squares(residual, [1.6, 1.7, 1.1], bounds=(0., np.inf)).x
    _coeff = terminal_map(_params, H_TARGET)

def predict(row):
    H = float(row['H'])
    coeff = _coeff if H == H_TARGET else terminal_map(_params, H)
    return float(coeff @ np.array([row[k] for k in KEYS]))
