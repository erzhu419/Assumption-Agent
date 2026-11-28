"""Panel-calibrated trajectory and scalar readouts."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibrated(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.linalg.norm(v, axis=1) < 4
    u, v = u[keep], v[keep]
    w = np.roll(u, 2, axis=1)

    def profile(p, coefficients=False):
        h = u + p[0] * np.arctan(p[1] * u)
        r = np.sin(p[2] * w)
        hc, rc, vc = h - h.mean(0), r - r.mean(0), v - v.mean(0)
        denom = (hc**2).sum(0)
        g0 = (hc * vc).sum(0) / denom
        g1 = (hc * rc).sum(0) / denom
        rr = rc - hc * g1
        cross = (rr * (vc - hc * g0)).sum() / (rr**2).sum()
        gain = g0 - cross * g1
        bias = (v - h * gain - cross * r).mean(0)
        return (bias, gain, cross) if coefficients else (bias + h * gain + cross * r - v).ravel()

    p = least_squares(profile, [.26, 1.75, 2.4],
                      bounds=([0, .3, 1], [1, 5, 4]), gtol=1e-13).x
    bias, gain, cross = profile(p, True)

    def forward(z):
        return bias + gain * (z + p[0] * np.arctan(p[1] * z)) + cross * np.sin(p[2] * np.roll(z, 2))

    return np.asarray([least_squares(lambda z: forward(z) - row, (row - bias) / gain,
                                    gtol=1e-13).x for row in x])


def _operator(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def warp(x, p):
        return np.sign(x) * np.maximum(np.abs(x) - p[0], 0.)**p[1]

    def profile(p):
        h = _affine(warp(u, p))
        return (h @ _fit(h, v) - v).ravel()

    p = least_squares(profile, [.35, 1.5], bounds=([0, .2], [.7, 4]), gtol=1e-13).x
    q = warp(z, p)
    return q, _affine(q) @ _fit(_affine(warp(u, p)), v)


def _designs(study, x):
    z = _calibrated(study['panel_a'], x)
    q, b = _operator(study['panel_b'], z)
    interaction = np.sum(b * np.roll(z, -1, axis=1), axis=1)
    state = _affine(np.column_stack((b, interaction)))
    # A scalar-only cross-gate couples the threshold response to panel B.
    scalar = np.column_stack((state, q / (1 + .5 * z[:, 1:2]**2),
                              q * np.sin(2.15 * b[:, 2:3])))
    return scalar, state


def solve_batch(study: dict) -> dict:
    """Fit each readout using only the labeled training rows."""
    tr = study['train']
    n = len(tr['x'])
    scalar, state = _designs(study, np.concatenate((tr['x'], study['test']['x'])))
    y = scalar[n:] @ _fit(scalar[:n], tr['y'])
    trajectory = state[n:] @ _fit(state[:n], tr['traj'])
    terminal = trajectory[:, -1]
    return {'y_hat': y, 'state_pred': trajectory,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
