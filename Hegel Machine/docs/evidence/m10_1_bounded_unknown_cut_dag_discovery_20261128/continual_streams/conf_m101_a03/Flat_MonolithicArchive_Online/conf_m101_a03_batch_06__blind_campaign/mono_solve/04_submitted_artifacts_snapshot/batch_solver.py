import numpy as np
from scipy.optimize import least_squares, root


_PAIRS = np.triu_indices(4)


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    i, j = _PAIRS
    return np.column_stack((_affine(x), x[:, i] * x[:, j]))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibration(panel, observed):
    u, v = panel['u'], panel['v']
    next_u = np.roll(u, -1, axis=1)
    cross = (1 + u) * next_u
    cube = u**3

    def fit(rows):
        # Separate per-channel offset and gain; pool the cyclic nonlinear terms.
        identity = np.eye(4)[None] + np.zeros((len(u), 1, 1))
        design = np.concatenate((identity, identity * u[:, :, None],
                                 cross[:, :, None], cube[:, :, None]), axis=2)
        return _fit(design[rows].reshape(-1, 10), v[rows].ravel())

    def forward(z, p):
        return p[:4] + p[4:8]*z + p[8]*(1+z)*np.roll(z, -1, axis=-1) + p[9]*z**3

    keep = np.argsort(np.linalg.norm(v-u, axis=1))[:len(u)-3]
    p = fit(keep)
    keep = np.argsort(np.linalg.norm(forward(u, p)-v, axis=1))[:len(u)-3]
    p = fit(keep)
    return np.array([root(lambda z: forward(z, p)-row, row, tol=1e-10).x
                     for row in observed])


def _bounded_fit(z, y, kind='tanh', starts=20):
    q, a = _quadratic(z), _affine(z)
    basis = np.linalg.qr(q, mode='reduced')[0]
    target = y - basis @ (basis.T @ y)
    if kind == 'rat':
        f = lambda h: h/(1+h*h)
    elif kind == 'sin':
        f = np.sin
    else:
        f = np.tanh

    def residual(p):
        v = f(a @ p)
        v -= basis @ (basis.T @ v)
        return target - v * ((target @ v)/(v @ v + 1e-16))

    rng = np.random.default_rng(51)
    initial = np.row_stack(([0, 0, -1, 0, 1], np.eye(5)[1:],
                            rng.normal(size=(starts, 5))))
    best = min((least_squares(residual, seed, bounds=(-8, 8), max_nfev=120)
                for seed in initial), key=lambda result: result.cost)
    p = best.x
    return p, _fit(np.column_stack((q, f(a @ p))), y), f


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    z = _calibration(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    q_test = _quadratic(z[n:])
    p, w, f = _bounded_fit(z[:n], train['y'])
    y_hat = np.column_stack((q_test, f(_affine(z[n:]) @ p))) @ w
    state_pred = q_test @ _fit(_quadratic(z[:n]), train['traj'])
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
