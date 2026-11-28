"""Predict the scalar readout and two-mode trajectory from calibrated inputs."""

import numpy as np
from scipy.optimize import least_squares


def _design(*columns):
    return np.column_stack((np.ones(len(columns[0])), *columns))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    n, d = u.shape
    neighbor = np.roll(u, -1, axis=1)
    # Offset and gain are channel-specific; cyclic coupling is shared.
    eye = np.eye(d)
    matrix = np.concatenate((np.broadcast_to(eye, (n, d, d)),
                             u[:, :, None] * eye,
                             np.stack((neighbor, u * neighbor, u**3), axis=-1)), axis=-1)
    keep = np.ones(n, dtype=bool)
    for _ in range(8):
        w = _fit(matrix[keep].reshape(-1, 2*d + 3), v[keep].ravel())
        error = np.linalg.norm(matrix @ w - v, axis=1)
        trimmed = error <= max(1e-3, 4 * np.median(error))
        if trimmed.sum() < n // 2 or np.array_equal(trimmed, keep):
            break
        keep = trimmed
    bias, gain = w[:d], w[d:2*d]
    link, interaction, cubic = w[-3:]
    j = np.arange(d)

    def forward(z):
        r = np.roll(z, -1)
        return bias + gain*z + link*r + interaction*z*r + cubic*z**3

    def jacobian(z):
        jac = np.diag(gain + interaction*np.roll(z, -1) + 3*cubic*z*z)
        jac[j, (j+1) % d] = link + interaction*z
        return jac

    return np.array([least_squares(lambda z: forward(z)-row, (row-bias)/gain,
                                   jac=jacobian, gtol=1e-12).x for row in observed]).reshape(-1, d)


def _operators(panel, z):
    u, v1, v2 = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1', 'v2'))

    def pairs(a):
        return _design(a[:, 0]*a[:, 1], a[:, 0]*a[:, 3],
                       a[:, 1]*a[:, 2], a[:, 2]*a[:, 3])

    def cosine(a, omega):
        return _design(np.cos(omega * a))

    def residual(omega):
        design = cosine(u, omega[0])
        return (design @ _fit(design, v2) - v2).ravel()

    omega = least_squares(residual, [2.4], bounds=([1.8], [3.0])).x[0]
    p = pairs(z) @ _fit(pairs(u), v1)
    r = cosine(z, omega) @ _fit(cosine(u, omega), v2)
    return p, r


def _trajectory(fast, slow, times, measured):
    n = len(measured)
    nf = fast.shape[1]
    target = measured.ravel()

    def temporal(rates):
        return -np.expm1(-times[:, None] * rates[:2])

    def design(params, f, s):
        h = temporal(params)
        return np.concatenate((f[:, None, :] * h[None, :, 0, None],
                               s[:, None, :] * (h[:, 1] + params[2]*h[:, 0])[None, :, None]),
                              axis=2).reshape(len(f)*len(times), -1)

    def residual(params):
        a = design(params, fast[:n], slow[:n])
        return a @ _fit(a, target) - target

    params = least_squares(residual, [1.8, 0.42, 0.10],
                           bounds=([1.3, 0.15, -0.4], [2.5, 0.8, 0.5])).x
    coef = _fit(design(params, fast[:n], slow[:n]), target)
    slow_amp = slow[n:] @ coef[nf:]
    fast_amp = fast[n:] @ coef[:nf] + params[2]*slow_amp
    return np.column_stack((fast_amp, slow_amp)) @ temporal(params).T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    observed = np.concatenate((train['x'], study['test']['x']), axis=0)
    z = _calibrate(study['panel_a'], observed)
    p, r = _operators(study['panel_c'], z)
    dot = np.einsum('ij,ij->i', p, r)
    cross = p[:, 0] * r[:, 1]
    scalar = _design(p, r, dot)
    y = scalar[n:] @ _fit(scalar[:n], np.asarray(train['y'], dtype=float))
    state = _trajectory(_design(p, dot), _design(r, cross),
                        np.asarray(study['eval_times'], dtype=float),
                        np.asarray(train['traj'], dtype=float))
    final = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), final.mean(), final.std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
