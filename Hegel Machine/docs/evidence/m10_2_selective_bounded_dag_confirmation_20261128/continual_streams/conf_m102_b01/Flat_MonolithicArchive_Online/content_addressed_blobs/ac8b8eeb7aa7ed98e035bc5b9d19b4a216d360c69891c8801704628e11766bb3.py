"""Panel-calibrated prediction with separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, root


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _cal_basis(u):
    return np.stack((np.ones_like(u), u, np.tanh(1.8*u),
                     np.sin(2.4*np.roll(u, 2, axis=-1))), axis=-1)


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    basis = _cal_basis(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        coef = np.stack([_ls(basis[keep, j], v[keep, j]) for j in range(4)])
        err = np.linalg.norm((basis*coef).sum(axis=-1)-v, axis=1)
        med = np.median(err)
        new_keep = err <= max(8*med, 0.002)
        if np.array_equal(new_keep, keep) or new_keep.sum() < 8:
            break
        keep = new_keep
    coef = np.stack([_ls(basis[keep, j], v[keep, j]) for j in range(4)])
    z = np.empty_like(x, dtype=float)
    for i, row in enumerate(x):
        def residual(q):
            return (_cal_basis(q)*coef).sum(axis=-1)-row
        guess = (row-coef[:, 0])/coef[:, 1]
        opt = root(residual, guess, tol=1e-10)
        z[i] = opt.x if np.linalg.norm(residual(opt.x)) < 1e-8 else least_squares(residual, guess).x
    return z


def _latent(study, x):
    z = _calibrate(study['panel_a'], x)
    b, c = study['panel_b'], study['panel_c']
    def b_design(u):
        return _aff(np.cos(2.6*u))
    def c_design(u):
        return _aff(np.tanh(1.75*u)+0.30*np.sin(2.1*np.roll(u, 1, axis=-1)))
    w = b_design(z) @ _ls(b_design(b['u']), b['v'])
    return c_design(w) @ _ls(c_design(c['u']), c['v1'])


def _trajectory(train_design, test_design, times, observed):
    # Regress the two spatial amplitudes and constrain their time dependence.
    spatial_fit = _ls(train_design, observed)
    fitted = (train_design @ spatial_fit).T
    def basis(rates):
        return -np.expm1(-times[:, None]*rates)
    def residual(rates):
        b = basis(rates)
        return (b @ _ls(b, fitted)-fitted).ravel()
    rates = least_squares(residual, [0.4, 1.8], bounds=([0.02, 1.0], [0.99, 4.0])).x
    b = basis(rates)
    return (test_design @ _ls(b, spatial_fit.T).T) @ b.T


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    q = _latent(study, np.concatenate((tr['x'], study['test']['x'])))
    a = _aff(q)
    d = np.column_stack((a, q[:, 0]*q[:, 2], q[:, 1]*q[:, 3]))
    y = d[n:] @ _ls(d[:n], tr['y'])
    state = _trajectory(a[:n], a[n:], np.asarray(study['eval_times']), tr['traj'])
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
