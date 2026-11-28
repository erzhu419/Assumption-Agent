"""Calibrated panel-C operators with separate scalar and two-rate state readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

def _design(*x):
    return np.column_stack((np.ones(len(x[0])), *x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibrate(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    mask = np.max(np.abs(u-v), axis=1) < 3
    u, v = u[mask], v[mask]
    j = np.arange(4)
    eye = np.broadcast_to(np.eye(4), (len(u), 4, 4))
    base = (eye, eye*u[:, :, None])

    def profile(rate):
        a = np.concatenate((*base, np.tanh(rate[0]*u)[:, :, None],
                            np.sin(rate[1]*np.roll(u, 2, axis=1))[:, :, None]), axis=2).reshape(-1, 10)
        w = _fit(a, v.ravel())
        return a @ w-v.ravel(), w

    k = least_squares(lambda p: profile(p)[0], [1.8, 2.4], bounds=([.5, 1.5], [4., 3.3])).x
    _, w = profile(k)
    offset, gain, a, b = w[:4], w[4:8], w[8], w[9]
    z = (observed-offset)/gain
    # Fixed-point warm start; refine against the original equations below.
    for _ in range(55):
        update = (observed-offset-a*np.tanh(k[0]*z)-b*np.sin(k[1]*np.roll(z, 2, axis=1)))/gain
        if np.max(np.abs(update-z), initial=0.) < 1e-12:
            break
        z = update
    # Newton refinement (the self term is monotone and dominates the coupling).
    for _ in range(12):
        r = np.roll(z, 2, axis=1)
        h = np.tanh(k[0]*z)
        err = offset+gain*z+a*h+b*np.sin(k[1]*r)-observed
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z),4,4))
        jac[:,j,j] = gain+a*k[0]*(1-h*h)
        jac[:,j,(j+2)%4] = b*k[1]*np.cos(k[1]*r)
        z -= np.linalg.solve(jac, err[...,None])[...,0]
    return z


_PAIRS = ((0, 1), (0, 3), (1, 2), (2, 3))


def _operators(panel, z):
    u = np.asarray(panel['u'], dtype=float)
    v1 = np.asarray(panel['v1'], dtype=float)
    v2 = np.asarray(panel['v2'], dtype=float)

    def pair_design(x):
        return _design(*(x[:, i] * x[:, j] for i, j in _PAIRS))

    def cosine_design(x, rate):
        return _design(np.cos(rate * x))

    def panel_error(rate):
        phi = cosine_design(u, rate)
        return np.sum((phi @ _fit(phi, v2) - v2)**2)

    rate = minimize_scalar(panel_error, bounds=(2.0, 2.8), method='bounded',
                           options={'xatol': 1e-10}).x
    p = pair_design(z) @ _fit(pair_design(u), v1)
    r = cosine_design(z, rate) @ _fit(cosine_design(u, rate), v2)
    return p, r


def _basis(times, rates):
    return -np.expm1(-times[:, None] * rates)


def _fit_state(times, observed, fast, slow):
    n = len(observed)
    target = observed.ravel()

    def profile(rates):
        h = _basis(times, rates)
        blocks = [f[:n, None, :] * h[None, :, j, None]
                  for j, f in enumerate((fast, slow))]
        a = np.concatenate(blocks, axis=2).reshape(target.size, -1)
        w = _fit(a, target)
        return a @ w - target, w

    rates = least_squares(lambda k: profile(k)[0], [1.8, 0.43],
                         bounds=([0.8, 0.03], [5.0, 1.0])).x
    _, w = profile(rates)
    k = fast.shape[1]
    amplitudes = np.column_stack((fast[n:] @ w[:k], slow[n:] @ w[k:]))
    return amplitudes @ _basis(times, rates).T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], np.asarray(x, dtype=float))
    p, r = _operators(study['panel_c'], z)
    common = _design(p, r, np.sum(p * r, axis=1))
    # Both state modes use the directional interaction; the scalar readout does not.
    cross = p[:, 0] * r[:, 1]
    fast = _design(common[:, 1:], cross)
    slow = _design(r, cross)
    y_hat = common[n:] @ _fit(common[:n], np.asarray(train['y'], dtype=float))
    state = _fit_state(np.asarray(study['eval_times'], dtype=float),
                       np.asarray(train['traj'], dtype=float), fast, slow)
    terminal = state[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state, 'diagnostics_vec': diagnostics}
