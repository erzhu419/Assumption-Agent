"""Recover the latent coordinates and fit scalar and relaxation readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = panel['u'], panel['v']
    neighbor = np.roll(u, -1, axis=1)
    # Channel-specific intercepts and gains; shared cyclic and cubic terms.
    indicators = np.broadcast_to(np.eye(4), (len(u), 4, 4))
    design = np.concatenate((indicators, indicators * u[..., None],
                             np.stack((neighbor, u * neighbor, u**3), axis=-1)), axis=-1)
    matrix = design.reshape(-1, design.shape[-1])
    start = _fit(matrix, v.ravel())
    robust = least_squares(lambda w: matrix @ w - v.ravel(), start,
                           jac=lambda w: matrix.copy(), loss='cauchy', f_scale=.01).x
    error = np.linalg.norm(design @ robust - v, axis=1)
    keep = error < 8 * np.median(error)
    return _fit(design[keep].reshape(-1, design.shape[-1]), v[keep].ravel())


def _invert(raw, coefficients):
    offset, gain = coefficients[:4], coefficients[4:8]
    coupling, interaction, cubic = coefficients[8:]
    z = (raw - offset) / gain
    j = np.arange(4)
    jacobian = np.zeros((len(z), 4, 4))
    for _ in range(15):
        neighbor = np.roll(z, -1, axis=1)
        error = offset + gain*z + coupling*neighbor + interaction*z*neighbor + cubic*z**3 - raw
        if np.max(np.abs(error), initial=0.) < 1e-12:
            break
        jacobian[:, j, j] = gain + interaction*neighbor + 3*cubic*z*z
        jacobian[:, j, (j+1) % 4] = coupling + interaction*z
        z -= np.linalg.solve(jacobian, error[..., None])[..., 0]
    return z


def _scalar(b):
    return np.column_stack((_affine(b), b[:, 0]*b[:, 2], b[:, 1]*b[:, 3]))


def _trajectory(x, xt, times, observations):
    # Fit the temporal modes to the affine spatial signal, not observation noise.
    q, r = np.linalg.qr(_affine(x), mode='reduced')
    signal = q.T @ observations

    def modes(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        basis = modes(rates)
        return (basis @ _fit(basis, signal.T) - signal.T).ravel()

    rates = least_squares(residual, [1.7, .4],
                          bounds=([.7, .02], [5., .68])).x
    basis = np.linalg.qr(modes(rates), mode='reduced')[0]
    return (_affine(xt) @ np.linalg.solve(r, signal @ basis)) @ basis.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibration(study['panel_a'])
    x = _invert(train['x'], calibration)
    xt = _invert(study['test']['x'], calibration)
    panel = study['panel_b']
    mixing = _fit(_affine(panel['u']), panel['v'])
    b, bt = _affine(x) @ mixing, _affine(xt) @ mixing
    y_hat = _scalar(bt) @ _fit(_scalar(b), train['y'])
    state_pred = _trajectory(x, xt, study['eval_times'], train['traj'])
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
