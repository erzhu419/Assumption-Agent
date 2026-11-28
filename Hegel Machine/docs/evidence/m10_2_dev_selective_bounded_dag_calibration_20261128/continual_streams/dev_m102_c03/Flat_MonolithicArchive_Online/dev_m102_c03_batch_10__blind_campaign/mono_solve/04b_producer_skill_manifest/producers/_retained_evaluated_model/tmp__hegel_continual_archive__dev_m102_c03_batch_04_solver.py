"""Calibrated bilinear readout of two measured operators, with shared relaxation."""
import numpy as np
from scipy.optimize import least_squares

_CYCLE = ((0, 1), (0, 3), (1, 2), (2, 3))


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        um, vm = u[keep].mean(0), v[keep].mean(0)
        du = u[keep] - um
        gain = np.sum(du * (v[keep] - vm), axis=0) / np.sum(du**2, axis=0)
        offset = vm - gain * um
        err = np.linalg.norm(v - offset - u * gain, axis=1)
        new = err <= max(0.001, 3 * np.median(err))
        if new.sum() < 8 or np.array_equal(new, keep):
            break
        keep = new
    return offset, gain


def _cycle(z):
    return _affine(np.column_stack([z[:, i] * z[:, j] for i, j in _CYCLE]))


def _design(study, x):
    offset, gain = _calibration(study['panel_a'])
    z = (x - offset) / gain
    panel = study['panel_c']
    u, v1, v2 = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1', 'v2'))
    c = _cycle(z) @ _fit(_cycle(u), v1)

    def cosine(a, w):
        return _affine(np.cos(w[0] * a))

    def residual(w):
        h = cosine(u, w)
        return (h @ _fit(h, v2) - v2).ravel()

    w = least_squares(residual, [2.4], bounds=(1.5, 3.2)).x
    d = cosine(z, w) @ _fit(cosine(u, w), v2)
    cd = (c[:, :, None] * d[:, None, :]).reshape(len(z), 16)
    return np.column_stack((_affine(z), c, d, cd))


def _basis(times, traj):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        q = basis(rates)
        return (q @ _fit(q, traj.T) - traj.T).ravel()

    rates = least_squares(residual, [0.42, 1.83], bounds=([0.12, 1.0], [0.9, 4.0])).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    f = _design(study, np.vstack((train['x'], study['test']['x'])))
    q = _basis(np.asarray(study['eval_times']), np.asarray(train['traj']))
    # The scalar readout uses the diagonal contraction of the two branches.
    # Trajectories also use one off-diagonal interaction, selected on training data.
    traj = np.asarray(train['traj'])
    cross = f[:, 13:]
    diagonal = np.array([0, 5, 10, 15])
    off_diagonal = np.setdiff1d(np.arange(16), diagonal)
    screening = _fit(f[:n], traj)[13:]
    interaction = off_diagonal[np.argmax(np.linalg.norm(screening[off_diagonal], axis=1))]
    scalar = np.column_stack((f[:, 0], f[:, 5:13], cross[:, diagonal].sum(axis=1)))
    spatial = np.column_stack((scalar, cross[:, interaction]))
    y = scalar[n:] @ _fit(scalar[:n], train['y'])
    amplitudes = _fit(q, traj.T).T
    state = spatial[n:] @ _fit(spatial[:n], amplitudes) @ q.T
    last = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), last.mean(), last.std()]))
