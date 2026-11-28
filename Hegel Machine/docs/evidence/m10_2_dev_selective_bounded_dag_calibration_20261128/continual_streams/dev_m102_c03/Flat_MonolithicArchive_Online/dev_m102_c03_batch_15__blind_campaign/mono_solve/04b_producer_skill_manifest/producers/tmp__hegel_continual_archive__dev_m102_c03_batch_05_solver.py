"""Predict the dual-branch readout after inverting cyclic sensor calibration."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    rows = np.arange(4)
    nxt = (rows + 1) % 4

    def design(w):
        q = w[:, nxt]
        return np.stack((np.ones_like(w), w, q + w*q, w**3), axis=-1)

    a = design(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(12):
        co = np.column_stack([_fit(a[keep, j], v[keep, j]) for j in rows])
        err = np.linalg.norm(np.einsum('njk,kj->nj', a, co) - v, axis=1)
        new = err <= max(0.001, 3 * np.median(err))
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new
    # Pool the two shared nonlinear terms after the robust row screening.
    ident = np.broadcast_to(np.eye(4), (len(u), 4, 4))
    g = np.concatenate((ident, ident*u[:, :, None], a[:, :, 2:]), axis=2)
    off, gain, common = np.split(_fit(g[keep].reshape(-1, 10), v[keep].ravel()), [4, 8])
    coupling, cubic = common
    lin = np.diag(gain)
    lin[rows, nxt] = coupling
    z0 = np.linalg.solve(lin, (x - off).T).T

    def forward(z):
        return off + gain*z + coupling*z[nxt]*(1+z) + cubic*z**3

    def jac(z):
        result = np.diag(gain + coupling*z[nxt] + 3*cubic*z*z)
        result[rows, nxt] = coupling*(1+z)
        return result

    return np.array([least_squares(lambda z: forward(z)-target, p, jac=jac,
                                   gtol=1e-12).x for target, p in zip(x, z0)])


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    u, v1, v2 = (np.asarray(study['panel_c'][k], dtype=float) for k in ('u', 'v1', 'v2'))

    def cycle(q):
        return _aff(q * np.roll(q, -1, axis=1))

    c = cycle(z) @ _fit(cycle(u), v1)

    def cosine(q, rate):
        return _aff(np.cos(q*rate[0]))

    def residual(rate):
        a = cosine(u, rate)
        return (a @ _fit(a, v2) - v2).ravel()

    rate = least_squares(residual, [2.4], bounds=(1.5, 3.2), gtol=1e-12).x
    d = cosine(z, rate) @ _fit(cosine(u, rate), v2)
    cross = (c[:, :, None]*d[:, None, :]).reshape(len(z), 16)
    return z, c, d, cross


def _basis(times, traj):
    def modes(rates):
        return -np.expm1(-times[:, None]*rates)

    def residual(rates):
        q = modes(rates)
        return (q @ _fit(q, traj.T) - traj.T).ravel()

    rates = least_squares(residual, [.42, 1.83], bounds=([.12, 1.0], [.9, 4.0])).x
    return modes(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    z, c, d, cross = _features(study, x)
    traj = np.asarray(train['traj'])
    q = _basis(np.asarray(study['eval_times']), traj)
    amplitudes = _fit(q, traj.T).T
    diag = np.arange(0, 16, 5)
    offdiag = np.setdiff1d(np.arange(16), diag)
    scalar = np.column_stack((_aff(c), d, cross[:, diag].sum(axis=1)))
    y = scalar[n:] @ _fit(scalar[:n], train['y'])
    # The trajectory has an additional off-diagonal branch interaction.
    full = np.column_stack((scalar, cross[:, offdiag]))
    coef = _fit(full[:n], amplitudes)
    chosen = offdiag[np.argmax(np.linalg.norm(coef[scalar.shape[1]:], axis=1))]
    spatial = np.column_stack((scalar, cross[:, chosen]))
    state = spatial[n:] @ _fit(spatial[:n], amplitudes) @ q.T
    last = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), last.mean(), last.std()]))
