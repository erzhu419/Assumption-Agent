"""Fit a paired nonlinear calibration, then predict dual-branch responses."""
import numpy as np
from scipy.optimize import least_squares


def _ols(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _design(z):
    return np.column_stack((np.ones(len(z)), z))


def _soft(z):
    return z / (1.0 + np.abs(z))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    affine = _design(u)
    keep = np.arange(len(u))
    for _ in range(6):
        linear = _ols(affine[keep], v[keep])
        err = np.linalg.norm(affine @ linear - v, axis=1)
        med = np.median(err)
        mad = np.median(np.abs(err - med))
        nxt = np.flatnonzero(err <= max(0.015, med + 7.0 * 1.4826 * mad))
        if len(nxt) < 12 or np.array_equal(nxt, keep):
            break
        keep = nxt
    ug, vg = u[keep], v[keep]

    def design(z, j, shape):
        a, b = z[:, j], z[:, (j + 2) % 4]
        return _design(np.column_stack((a, _soft(shape[0] * a), b, np.tanh(shape[1] * b))))

    def residual(shape):
        residuals = []
        for j in range(4):
            d = design(ug, j, shape)
            residuals.append(d @ _ols(d, vg[:, j]) - vg[:, j])
        return np.concatenate(residuals)

    shape = least_squares(residual, [0.4, 1.0], bounds=(0.08, 8.0), gtol=1e-12).x
    coef = np.column_stack([_ols(design(ug, j, shape), vg[:, j]) for j in range(4)])
    a, b = shape
    col = np.arange(4)
    opp = (col + 2) % 4

    def inverse(x):
        x = np.asarray(x, dtype=float)
        z = (x - linear[0]) @ np.linalg.pinv(linear[1:])

        def model(z):
            w = z[..., opp]
            return coef[0] + coef[1] * z + coef[2] * _soft(a * z) + coef[3] * w + coef[4] * np.tanh(b * w)

        def jac(z):
            j = np.zeros(z.shape[:-1] + (4, 4))
            j[..., col, col] = coef[1] + coef[2] * a / (1.0 + np.abs(a * z))**2
            j[..., col, opp] = coef[3] + coef[4] * b * (1.0 - np.tanh(b * z[..., opp])**2)
            return j

        for _ in range(16):
            r = model(z) - x
            if not len(z) or np.max(np.abs(r)) <= 1e-11:
                break
            step = (np.linalg.pinv(jac(z)) @ r[..., None])[..., 0]
            z -= np.clip(step, -0.5, 0.5)
        for i in np.flatnonzero(np.max(np.abs(model(z) - x), axis=1) > 1e-7):
            z[i] = least_squares(lambda p: model(p) - x[i], z[i], jac=jac).x
        return z

    return inverse


def _branches(panel):
    u = np.asarray(panel['u'], dtype=float)

    def fit(v, transform, start):
        def design(z, scale):
            return _design(np.column_stack((z, transform(scale[0] * z))))

        def residual(scale):
            d = design(u, scale)
            return (d @ _ols(d, v) - v).ravel()

        scale = least_squares(residual, [start], bounds=(0.05, 4.0), gtol=1e-13).x
        coef = _ols(design(u, scale), v)
        return lambda z: design(z, scale) @ coef

    return (fit(panel['v1'], _soft, 0.84),
            fit(panel['v2'], lambda z: 1.0 / (1.0 + z*z), 0.4))


def _features(study, x):
    z = _calibration(study['panel_a'])(x)
    branch1, branch2 = _branches(study['panel_c'])
    c, d = branch1(z), branch2(z)
    outer = (c[:, :, None] * d[:, None, :]).reshape(len(x), 16)
    base = np.column_stack((c, d))
    return (_design(np.column_stack((base, np.sum(c*d, axis=1)))),
            _design(np.column_stack((base, outer))))


def _trajectory(t, observed, train_features, test_features):
    def basis(rates):
        return -np.expm1(-t[:, None] * rates)

    def residual(rates):
        h = basis(rates)
        return (h @ _ols(h, observed.T) - observed.T).ravel()

    rates = least_squares(residual, [1.8, 0.4], bounds=([0.65, 0.025], [6.0, 0.65]), gtol=1e-11).x
    h = basis(rates)
    amplitudes = _ols(h, observed.T).T
    return (test_features @ _ols(train_features, amplitudes)) @ h.T


def solve_batch(study: dict) -> dict:
    """Return scalar predictions, time-grid predictions, and population moments."""
    tr = study['train']
    n = len(tr['x'])
    fy, ft = _features(study, np.vstack((tr['x'], study['test']['x'])))
    y = fy[n:] @ _ols(fy[:n], np.asarray(tr['y']).reshape(n))
    state = _trajectory(np.asarray(study['eval_times']), np.asarray(tr['traj']), ft[:n], ft[n:])
    terminal = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()]) if len(y) else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
