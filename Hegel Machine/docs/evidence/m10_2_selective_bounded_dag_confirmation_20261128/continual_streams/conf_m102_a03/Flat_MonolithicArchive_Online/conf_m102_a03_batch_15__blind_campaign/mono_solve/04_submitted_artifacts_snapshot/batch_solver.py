"""Calibrated cosine operator with separate scalar and time-dependent readouts."""
import numpy as np
from scipy.optimize import least_squares
def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = u.ravel()
    b = np.roll(u, 2, axis=1).ravel()
    channel = np.tile(np.eye(4), (len(u), 1))

    def design(p):
        return np.column_stack((channel, channel * a[:, None],
                                np.tanh(p[0] * a), np.sin(p[1] * b)))

    discrepancy = np.linalg.norm(v - u, axis=1)
    mask = discrepancy <= 4 * max(np.median(discrepancy), 1e-8)
    if mask.sum() < 8:
        mask[:] = True
    p = np.array([1.8, 2.4])
    for _ in range(5):
        keep = np.repeat(mask, 4)
        target = v.ravel()[keep]

        def residual(q):
            f = design(q)[keep]
            return f @ _fit(f, target) - target

        p = least_squares(residual, p, bounds=([0.5, 1.5], [3.5, 3.5]),
                          ftol=1e-12, gtol=1e-12, xtol=1e-12).x
        f = design(p)
        c = _fit(f[keep], target)
        err = np.linalg.norm((f @ c).reshape(v.shape) - v, axis=1)
        updated = err <= max(6 * np.median(err), 1e-7)
        if updated.sum() < 8 or np.array_equal(updated, mask):
            break
        mask = updated
    return p, c


def _decode(x, cal):
    p, c = cal
    bias, gain = c[:4], c[4:8]
    h, k = c[8:]
    i = np.arange(4)
    j = (i + 2) % 4

    def residual(z, x):
        return bias + gain * z + h * np.tanh(p[0] * z) + k * np.sin(p[1] * z[j]) - x

    def jac(z, x):
        q = np.tanh(p[0] * z)
        g = np.diag(gain + h * p[0] * (1 - q*q))
        g[i, j] = k * p[1] * np.cos(p[1] * z[j])
        return g

    z = [least_squares(residual, (row - bias) / gain, jac=jac, args=(row,),
                       gtol=1e-12, ftol=1e-12, xtol=1e-12).x for row in x]
    return np.asarray(z).reshape(-1, 4)



def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _operator(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(q, freq):
        return _affine(np.cos(freq * q))
    def residual(p):
        a = design(u, p[0])
        return (a @ _fit(a, v) - v).ravel()
    freq = least_squares(residual, [2.6], bounds=(1.5, 3.8)).x[0]
    return design(z, freq) @ _fit(design(u, freq), v)


def _readouts(b):
    # The scalar has a shared coefficient on the two opposite-channel products.
    opposite_pairs = b[:, 0] * b[:, 2] + b[:, 1] * b[:, 3]
    return _affine(b), np.column_stack((b, opposite_pairs))


def _forecast_state(h, train_h, time, traj):
    spatial = _fit(train_h, traj)
    def basis(p):
        return -np.expm1(-np.outer(time, p))
    def residual(p):
        a = basis(p)
        pred = train_h @ _fit(a, spatial.T).T @ a.T
        return (pred - traj).ravel()
    rates = least_squares(residual, [0.41, 1.82], bounds=([0.05, 0.85], [0.83, 5.])).x
    a = basis(rates)
    return h @ _fit(a, spatial.T).T @ a.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    z = _decode(np.concatenate((train['x'], study['test']['x'])), _calibration(study['panel_a']))
    h, q = _readouts(_operator(study['panel_b'], z))
    y_hat = q[n:] @ _fit(q[:n], train['y'])
    state_pred = _forecast_state(h[n:], h[:n], study['eval_times'], train['traj'])
    last = state_pred[:, -1]
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics_vec)
