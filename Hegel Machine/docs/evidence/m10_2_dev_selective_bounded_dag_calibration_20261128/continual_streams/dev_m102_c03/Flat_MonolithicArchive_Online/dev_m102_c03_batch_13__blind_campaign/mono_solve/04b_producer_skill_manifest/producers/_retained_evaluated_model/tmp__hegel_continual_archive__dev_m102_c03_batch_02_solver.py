"""Recover the calibrated dual-branch response and its relaxation modes."""
import numpy as np
from scipy.linalg import qr
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

    # Restrict each relaxation component to the measured branch interactions.
    cross = u[:, 0] * v[:, 1]
    diagonal = np.sum(u * v, axis=1)
    groups = (
        np.column_stack((_affine(v), cross)),
        np.column_stack((_affine(u), diagonal)),
        np.column_stack((v, cross)),
    )
    orthogonal = [qr(f[:n], mode='economic')[0] for f in groups]
    target = trajectories.ravel()

    def project(log_rates, predict=False):
        basis = -np.expm1(-np.exp(log_rates)[:, None] * times)
        design = np.column_stack([
            (a[:, None, :] * q[None, :, None]).reshape(target.size, -1)
            for a, q in zip(orthogonal, basis)
        ])
        weights = np.linalg.lstsq(design, target, rcond=1e-11)[0]
        if not predict:
            return design @ weights - target
        pred = np.zeros((len(x) - n, times.size))
        start = 0
        for f, a, q in zip(groups, orthogonal, basis):
            stop = start + a.shape[1]
            coeff = np.linalg.lstsq(f[:n], a @ weights[start:stop], rcond=1e-10)[0]
            pred += (f[n:] @ coeff)[:, None] * q[None, :]
            start = stop
        return pred

    rates = least_squares(
        project, np.log([0.42, 1.8, 0.65]),
        bounds=(np.log([0.12, 1.0, 0.08]), np.log([0.9, 3.8, 4.0])),
        max_nfev=80,
    ).x
    state_pred = project(rates, predict=True)
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])
    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics_vec)
