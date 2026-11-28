"""Invert the reference-panel calibration and fit the two downstream outputs."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    n = len(u)
    a = u.ravel()
    b = np.roll(u, -1, axis=1).ravel()
    channel = np.tile(np.eye(4), (n, 1))
    design = np.column_stack((channel, channel * a[:, None], b, a*b, a**3))
    discrepancy = np.linalg.norm(v-u, axis=1)
    good = discrepancy < 4 * max(np.median(discrepancy), 1e-8)
    if good.sum() < 8:
        good[:] = True
    for _ in range(5):
        keep = np.repeat(good, 4)
        coef = _fit(design[keep], v.ravel()[keep])
        residual = np.linalg.norm((design @ coef).reshape(n, 4) - v, axis=1)
        refined = residual <= max(5 * np.median(residual), 1e-7)
        if refined.sum() < 8 or np.array_equal(good, refined):
            break
        good = refined
    return coef


def _decode(x, coef):
    offset, gain = coef[:4], coef[4:8]
    coupling, interaction, cubic = coef[8:]
    i = np.arange(4)
    j = (i + 1) % 4

    def residual(z, row):
        nxt = z[j]
        return offset + gain*z + coupling*nxt + interaction*z*nxt + cubic*z**3 - row

    def jacobian(z, row):
        g = np.diag(gain + interaction*z[j] + 3*cubic*z*z)
        g[i, j] = coupling + interaction*z
        return g

    z = []
    for row in x:
        result = least_squares(residual, (row-offset)/gain, jac=jacobian,
                               args=(row,), gtol=1e-12, max_nfev=80)
        if np.linalg.norm(result.fun, np.inf) > 1e-8:
            other = least_squares(residual, np.zeros(4), jac=jacobian,
                                  args=(row,), gtol=1e-12, max_nfev=100)
            if np.linalg.norm(other.fun) < np.linalg.norm(result.fun):
                result = other
        z.append(result.x)
    return np.asarray(z).reshape(-1, 4)


def _readout_features(w):
    interaction = w[:, 0]*w[:, 2] + w[:, 1]*w[:, 3]
    return np.column_stack((w, interaction))


def _relaxation(w, traj, times):
    # Project onto the spatial design before fitting the shared time constants.
    q, r = np.linalg.qr(w, mode='reduced')
    target = q.T @ traj

    def basis(rates):
        return -np.expm1(-np.outer(rates, times))

    def residual(rates):
        f = basis(rates)
        return (_fit(f.T, target.T).T @ f - target).ravel()

    fit = least_squares(residual, [0.4, 1.8], bounds=(1e-4, 20.),
                        ftol=1e-11, gtol=1e-11, xtol=1e-11)
    f = basis(np.sort(fit.x))
    return _fit(r, _fit(f.T, target.T).T) @ f


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    coef = _calibration(study['panel_a'])
    z, zt = _decode(x, coef), _decode(test, coef)
    b = study['panel_b']
    operator = _fit(_affine(np.asarray(b['u'])), np.asarray(b['v']))
    w, wt = _affine(z) @ operator, _affine(zt) @ operator
    yh = _readout_features(wt) @ _fit(_readout_features(w), train['y'])
    sp = wt @ _relaxation(w, np.asarray(train['traj']), np.asarray(study['eval_times']))
    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])
    return {'y_hat': yh, 'state_pred': sp, 'diagnostics_vec': diag}
