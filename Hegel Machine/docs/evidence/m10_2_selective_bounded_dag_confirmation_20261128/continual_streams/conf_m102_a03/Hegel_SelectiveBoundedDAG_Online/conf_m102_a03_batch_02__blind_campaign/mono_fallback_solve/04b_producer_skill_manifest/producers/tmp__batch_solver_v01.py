"""Calibrated response-surface and relaxation model for the public campaign."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    j = u.shape[1]
    coeff = np.zeros((2, j))
    mats = [np.column_stack((np.ones(len(u)), u[:, k])) for k in range(j)]
    w = np.ones(len(u))
    for _ in range(28):
        for k, design in enumerate(mats):
            coeff[:, k] = _fit(design * w[:, None], v[:, k] * w)
        r = np.linalg.norm(v - coeff[0] - u * coeff[1], axis=1)
        scale = max(np.median(r), 1e-12)
        w = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(r, 1e-12)))
    keep = r < 6 * scale
    if np.count_nonzero(keep) >= 3:
        for k, design in enumerate(mats):
            coeff[:, k] = _fit(design[keep], v[keep, k])
    return coeff


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _quadratic(z):
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack((_affine(z), z[:, i] * z[:, j]))


def _readout(z, z_test, y):
    l, lt = _affine(z), _affine(z_test)
    q, qt = _quadratic(z), _quadratic(z_test)
    orth, _ = np.linalg.qr(q, mode='reduced')
    y_linear = _fit(q, y)
    target = y - q @ y_linear
    direction = _fit(l, y)[1:]
    direction /= max(np.linalg.norm(direction), 1e-12)
    rng = np.random.default_rng(98)
    starts = [np.r_[offset, scale * direction] for offset in (0., -.3)
              for scale in (.45, 1., 1.8, 3., 5.)]
    starts += [np.r_[0., d] for d in rng.normal(size=(8, z.shape[1]))]

    def fit_branch(mode):
        def features(a, h):
            g = a @ h
            if mode == 0:
                return (g / (1. + g*g))[:, None]
            if mode == 1:
                return a / (1. + g[:, None]**2)
            return a * expit(g[:, None])

        def residual(h):
            f = features(l, h)
            f = f - orth @ (orth.T @ f)
            return f @ _fit(f, target) - target

        fits = [least_squares(residual, start, bounds=(-12., 12.), max_nfev=100)
                for start in starts]
        h = min(fits, key=lambda v: v.cost).x
        design = np.column_stack((q, features(l, h)))
        return np.column_stack((qt, features(lt, h))) @ _fit(design, y)

    return .3 * fit_branch(0) + .4 * fit_branch(1) + .3 * fit_branch(2)


def _state(z, z_test, traj, times):
    # Profile amplitudes over shared time constants; keep the spatial law affine.
    l, lt = _affine(z), _affine(z_test)
    a = _fit(l, traj)
    qr, _ = np.linalg.qr(l, mode='reduced')
    score = qr.T @ traj

    def basis(h):
        return -np.expm1(-times[:, None] * np.exp(h))

    def residual(h):
        b = basis(h)
        return (b @ _fit(b, score.T) - score.T).ravel()

    fits = [least_squares(residual, np.log(start), bounds=(-5., 3.), max_nfev=100)
            for start in ((.3, 1.5), (.1, 3.), (.65, 2.5))]
    b = basis(min(fits, key=lambda r: r.cost).x)
    return lt @ (.15 * a + .85 * (_fit(b, a.T).T @ b.T))


def solve_batch(study: dict) -> dict:
    """Fit from the supplied training observations and predict the test rows."""
    train = study['train']
    shift, scale = _calibrate(study['panel_a'])
    z = (np.asarray(train['x'], dtype=float) - shift) / scale
    z_test = (np.asarray(study['test']['x'], dtype=float) - shift) / scale
    y_hat = _readout(z, z_test, np.asarray(train['y'], dtype=float))
    state_pred = _state(z, z_test, np.asarray(train['traj'], dtype=float),
                        np.asarray(study['eval_times'], dtype=float))
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
