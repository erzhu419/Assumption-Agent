"""Invert the cyclic calibration and fit scalar and relaxation readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = panel['u'], panel['v']
    rng = np.random.default_rng(123)
    rows = []
    for a, b, y in zip(u.T, np.roll(u, -1, axis=1).T, v.T):
        design = np.column_stack((np.ones(len(a)), a, b, a*b, a**3))
        best_loss = np.inf
        for start in range(48):
            keep = rng.choice(len(a), 6, replace=False) if start else np.arange(len(a))
            for _ in range(7):
                resid = np.abs(y - design @ _fit(design[keep], y[keep]))
                keep = np.argsort(resid)[:44]
            coef = _fit(design[keep], y[keep])
            resid = np.abs(y - design @ coef)
            loss = np.linalg.norm(resid[keep])
            if loss < best_loss:
                best_loss, best_resid = loss, resid
        keep = best_resid < 0.001
        rows.append(_fit(design[keep], y[keep]))
    return np.array(rows)


def _decode(x, calibration):
    offset, gain, coupling, product, cubic = calibration.T
    z = (x - offset) / gain
    ix = np.arange(z.shape[1])
    jac = np.zeros((len(z), len(ix), len(ix)))
    for _ in range(20):
        next_z = np.roll(z, -1, axis=1)
        resid = offset + gain*z + coupling*next_z + product*z*next_z + cubic*z**3 - x
        if np.max(np.abs(resid)) < 1e-12:
            break
        jac[:, ix, ix] = gain + product*next_z + 3*cubic*z*z
        jac[:, ix, (ix + 1) % len(ix)] = coupling + product*z
        z -= np.linalg.solve(jac, resid[..., None])[..., 0]
    return z


def _scalar_design(z):
    return np.column_stack((_affine(z), z[:, 0]*z[:, 2], z[:, 1]*z[:, 3]))


def _trajectory(times, observed, z, zt):
    def basis(rates):
        return -np.expm1(-times[:, None]*rates)

    def residual(rates):
        temporal = basis(rates)
        return (temporal @ _fit(temporal, observed.T) - observed.T).ravel()

    rates = least_squares(residual, [1.8, 0.4], bounds=([0.7, 0.03], [5.0, 0.68])).x
    temporal = basis(rates)
    amplitude = _fit(temporal, observed.T).T
    return (_affine(zt) @ _fit(_affine(z), amplitude)) @ temporal.T


def solve_batch(study: dict) -> dict:
    train, xt = study['train'], study['test']['x']
    calibration = _calibration(study['panel_a'])
    panel = study['panel_b']
    operator = _fit(_affine(panel['u']), panel['v'])
    z, zt = [_affine(_decode(x, calibration)) @ operator for x in (train['x'], xt)]
    y_hat = _scalar_design(zt) @ _fit(_scalar_design(z), train['y'])
    state_pred = _trajectory(study['eval_times'], train['traj'], z, zt)
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
