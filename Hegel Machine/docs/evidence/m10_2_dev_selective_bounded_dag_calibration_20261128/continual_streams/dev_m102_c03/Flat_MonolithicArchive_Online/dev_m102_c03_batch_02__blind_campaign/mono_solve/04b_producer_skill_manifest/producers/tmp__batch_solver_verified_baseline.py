"""Recover the calibrated dual-branch response and its two relaxation modes."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        um, vm = u[keep].mean(axis=0), v[keep].mean(axis=0)
        du = u[keep] - um
        gain = np.sum(du * (v[keep] - vm), axis=0) / np.sum(du**2, axis=0)
        offset = vm - gain * um
        err = np.linalg.norm(v - offset - u * gain, axis=1)
        new_keep = err <= max(0.001, 4 * np.median(err))
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return offset, gain


def _branches(panel, z):
    u, v1, v2 = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1', 'v2'))

    def odd(x, freq):
        q = x * freq
        return np.column_stack((_affine(x), q / (1.0 + np.abs(q))))

    def even(x, freq):
        return _affine(np.cos(x * freq))

    def project(feature, target, start, bounds):
        def linear(f):
            return np.linalg.lstsq(feature(u, f), target, rcond=1e-10)[0]

        def residual(f):
            return (feature(u, f) @ linear(f) - target).ravel()

        f = least_squares(residual, np.full(u.shape[1], start), bounds=bounds).x
        return feature(z, f) @ linear(f)

    return project(odd, v1, 0.83, (0.2, 3.0)), project(even, v2, 2.4, (1.5, 3.2))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    times = np.asarray(study['eval_times'], dtype=float)
    trajectories = np.asarray(train['traj'], dtype=float)
    offset, gain = _calibration(study['panel_a'])
    z = (x - offset) / gain
    u, v = _branches(study['panel_c'], z)
    main = np.column_stack((_affine(z), u, v))
    # The scalar response uses the exchange-symmetric diagonal contraction.
    scalar_features = np.column_stack((main, np.sum(u * v, axis=1)))
    y_hat = scalar_features[n:] @ np.linalg.lstsq(scalar_features[:n], train['y'], rcond=1e-9)[0]

    # The temporal channels can also use off-diagonal branch products.
    state_features = np.column_stack((main, (u[:, :, None] * v[:, None, :]).reshape(len(z), -1)))

    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates)[:, None] * times)

    def residual(log_rates):
        q = basis(log_rates)
        return (trajectories - trajectories @ np.linalg.pinv(q) @ q).ravel()

    q = basis(least_squares(residual, np.log([0.4, 1.8]), max_nfev=120).x)
    amplitudes = trajectories @ np.linalg.pinv(q)
    state_pred = (state_features[n:] @ np.linalg.lstsq(state_features[:n], amplitudes, rcond=1e-9)[0]) @ q
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])
    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics_vec)
