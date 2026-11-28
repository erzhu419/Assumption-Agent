import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    scale, bias = [], []
    for x, y in zip(u.T, v.T):
        dx = x[i] - x[j]
        nz = dx != 0
        m = np.median((y[i] - y[j])[nz] / dx[nz])
        b = np.median(y - m*x)
        err = np.abs(y - m*x - b)
        keep = err <= 5 * np.median(err)
        b, m = _fit(_affine(x[keep]), y[keep])
        scale.append(m)
        bias.append(b)
    return np.array(scale), np.array(bias)


def _activate(x, p):
    b, a, d, c = p
    return b*x + np.tanh(a*x) + d*np.tanh(c*x)


def _panel_map(panel):
    u, v = panel['u'], panel['v1']
    def residual(p):
        A = _affine(_activate(u, p))
        return (A @ _fit(A, v) - v).ravel()
    p = least_squares(residual, [.18, 1.56, .13, 6], bounds=([0, .5, 0, 2], [1, 3, 1, 12])).x
    W = _fit(_affine(_activate(u, p)), v)
    return lambda x: _affine(_activate(x, p)) @ W


def _readout(z):
    return np.column_stack((z, z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]))


def _time_basis(t, p):
    return -np.expm1(-t[:, None] * p)


def solve_batch(study):
    train, test = study['train'], study['test']
    m, b = _calibrate(study['panel_a'])
    panel_b = study['panel_b']
    W = _fit(_affine(panel_b['u']), panel_b['v'])
    panel_c = _panel_map(study['panel_c'])
    def encode(x):
        return panel_c(_affine((x - b) / m) @ W)
    z, zt = encode(train['x']), encode(test['x'])
    y = _readout(zt) @ _fit(_readout(z), train['y'])
    t = np.asarray(study['eval_times'])
    coeff = _fit(z, train['traj'])
    def residual(p):
        T = _time_basis(t, p)
        return (z @ (_fit(T, coeff.T).T @ T.T - coeff)).ravel()
    p = least_squares(residual, [.4, 1.8], bounds=([.05, 1], [1, 10])).x
    T = _time_basis(t, p)
    state = zt @ _fit(T, coeff.T).T @ T.T
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
