"""Robust affine calibration, two panel-identified branches, and exponential state modes."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = map(np.asarray, (panel['u'], panel['v']))
    coef = []
    for j in range(u.shape[1]):
        a = _aff(u[:, j:j+1])
        e = np.abs(a @ _ls(a, v[:, j]) - v[:, j])
        keep = e < max(.001, 4*np.median(e))
        for _ in range(3):
            fit = _ls(a[keep], v[keep, j])
            e = np.abs(a @ fit - v[:, j])
            good = e < max(.001, 4*np.median(e))
            if good.sum() >= 3:
                keep = good
        coef.append(_ls(a[keep], v[keep, j]))
    bias, gain = np.asarray(coef).T
    return (x-bias)/gain


def _branches(panel, z):
    u = np.asarray(panel['u'])
    out = []
    for k, initial in [(1, .66), (2, 1.2)]:
        v = np.asarray(panel[f'v{k}'])

        def design(a, rate):
            if k == 1:
                return np.column_stack((_aff(a), a**3, a/(1+rate*np.abs(a))))
            return np.column_stack((_aff(a), a*np.roll(a, -1, axis=1), a**3, np.tanh(rate*a)))

        def residual(rate):
            a = design(u, rate)
            return (a @ _ls(a, v) - v).ravel()

        rate = least_squares(residual, [initial], bounds=(.1, 3), gtol=1e-12).x
        out.append(design(z, rate) @ _ls(design(u, rate), v))
    return out


def _modes(times, trajectories):
    def basis(rates):
        return -np.expm1(-times[:, None]*rates)

    def residual(rates):
        q = basis(rates)
        return (q @ _ls(q, trajectories.T) - trajectories.T).ravel()

    rates = least_squares(residual, [.4, 1.85], bounds=([.1, .9], [.85, 4])).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    n = len(train['x'])
    z = _calibrate(study['panel_a'], x)
    b, c = _branches(study['panel_c'], z)
    interactions = (b[:, :, None]*c[:, None, :]).reshape(len(x), -1)
    y_design = _aff(np.column_stack((b, c, np.sum(b*c, axis=1))))
    state_design = _aff(np.column_stack((b, c, interactions)))
    q = _modes(np.asarray(study['eval_times']), np.asarray(train['traj']))
    y_hat = y_design[n:] @ _ls(y_design[:n], train['y'])
    amplitudes = _ls(q, np.asarray(train['traj']).T).T
    state_pred = state_design[n:] @ _ls(state_design[:n], amplitudes) @ q.T
    diagnostics = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
