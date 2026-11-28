import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist


def _quadratic(x):
    return np.column_stack([np.ones(len(x)), x] + [x[:, i] * x[:, j] for i in range(4) for j in range(i, 4)])


def _gp(x, z, y, length, nugget, matern=False):
    scale = x.std(axis=0)
    x, z = x / scale, z / scale
    def kernel(a, b):
        d = cdist(a, b, 'sqeuclidean') / length**2
        if not matern:
            return np.exp(-0.5 * d)
        r = np.sqrt(5 * d)
        return (1 + r + r*r/3) * np.exp(-r)
    f, g = _quadratic(x), _quadratic(z)
    K = kernel(x, x)
    K.flat[::len(x)+1] += nugget
    L = cholesky(K, lower=True)
    w = solve_triangular(L, np.column_stack([f, y]), lower=True)
    c = np.linalg.lstsq(w[:, :-1], w[:, -1], rcond=None)[0]
    return g @ c + kernel(z, x) @ solve_triangular(L.T, w[:, -1] - w[:, :-1] @ c)


def solve_batch(study):
    train = study['train']
    x, z = train['x'], study['test']['x']
    tr = train['traj']
    mode = np.linalg.svd(tr, full_matrices=False)[2][0]
    y = _gp(x, z, train['y'], 1.5, 0.0001)
    amp = _gp(x, z, tr @ mode, 2.4, 0.001, matern=True)
    state = amp[:, None] * mode
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
