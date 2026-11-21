import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row['raw_prefix']
    end = float(prefix[-1]['t'])
    z = 0.52 * float(prefix[0]['x']) * math.exp((float(prefix[0]['t']) - end) / tau)
    for left, right in zip(prefix, prefix[1:]):
        t0, t1 = float(left['t']), float(right['t'])
        f0 = float(left['x']) * math.exp((t0 - end) / tau)
        f1 = float(right['x']) * math.exp((t1 - end) / tau)
        z += (0.26 / tau) * (t1 - t0) * (f0 + f1)
    return z


def _readout(p, horizon):
    a, b, c, tau = p
    # Augmented state (x, z, u), with u constant. This polynomial is one RK4 step.
    generator = np.array([[-a, b, c], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    eye = np.eye(3)

    def step(h):
        q = h * generator
        return eye + q @ (eye + q @ (eye / 2 + q @ (eye / 6 + q / 24)))

    n = math.floor(horizon / 0.004)
    transition = np.linalg.matrix_power(step(0.004), n)
    remaining = horizon - n * 0.004
    if remaining > 0.0:
        transition = step(remaining) @ transition
    return transition[0]


def _evaluate(row, p, readout):
    return float(readout @ [float(row['x_now']), _memory(row, p[3]), float(row['u_now'])])


def fit(train_rows):
    global _params
    rows = list(train_rows)
    horizons = sorted({float(r['H']) for r in rows})
    anchor_fast_scale = len(horizons) == 1 and all(r.get('mem_conv') is not None for r in rows)
    targets = np.array([float(r['y']) for r in rows])

    def residual(p):
        readouts = {h: _readout(p, h) for h in horizons}
        errors = np.array([_evaluate(r, p, readouts[float(r['H'])]) for r in rows]) - targets
        if anchor_fast_scale:
            return np.append(errors, 0.01 * (p[3] - 0.25))
        return errors

    result = least_squares(residual, [2.0, 2.0, 1.2, 0.25], bounds=([0.0, 0.0, 0.0, 0.02], [15.0, 20.0, 20.0, 5.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x


def predict(row) -> float:
    return _evaluate(row, _params, _readout(_params, float(row['H'])))