import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    a = _affine(u)
    # Start with a high-breakdown residual screen before fitting the affine map.
    delta = v - u
    e0 = np.linalg.norm(delta - np.median(delta, axis=0), axis=1)
    initial = e0 <= max(1e-10, 6 * np.median(e0))
    mask = initial if initial.sum() >= max(u.shape[1] + 2, .75 * len(u)) else np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = _ls(a[mask], v[mask])
        e = np.linalg.norm(a @ w - v, axis=1)
        new = e <= max(1e-10, 4 * np.median(e))
        if new.sum() < 8 or np.array_equal(mask, new):
            break
        mask = new
    w = _ls(a[mask], v[mask])
    e = np.linalg.norm(a @ w - v, axis=1)
    d = np.array([_ls(a[mask][:, [0, j+1]], v[mask, j]) for j in range(4)]).T
    if np.median(np.linalg.norm((u*d[1]+d[0]-v)[mask], axis=1)) <= 1.35*np.median(e[mask])+1e-10:
        return d[0], np.diag(d[1])
    return w[0], w[1:]


def _panel_operators(panel):
    u = panel['u']
    def design(x, k, j):
        if j == 1:
            h = x / (1 + k * np.abs(x))
        else:
            h = x*x / (1 + k*x*x)
        return np.column_stack((_affine(x), h))
    def fit(j, initial):
        v = panel[f'v{j}']
        def residual(k):
            f = design(u, k[0], j)
            return (f @ _ls(f, v) - v).ravel()
        k = least_squares(residual, [initial], bounds=(0.01, 5.)).x[0]
        w = _ls(design(u, k, j), v)
        return lambda x: design(x, k, j) @ w
    return fit(1, 0.8), fit(2, 0.16)


def _features(p, r):
    # A common diagonal coefficient is represented by the dot product.
    common = np.column_stack((p, r, np.einsum('ij,ij->i', p, r)))
    offdiag = np.column_stack([p[:, i]*r[:, j] for i in range(4)
                               for j in range(4) if i != j])
    return common, offdiag


def _ridge(f, target, query, penalty):
    fm, ym = f.mean(axis=0), target.mean(axis=0)
    a = f-fm
    w = np.linalg.solve(a.T @ a + penalty*np.eye(f.shape[1]), a.T @ (target-ym))
    return (query-fm) @ w + ym


def _quadratic_design(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j]))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = train['x']
    combined = np.concatenate((x, study['test']['x']), axis=0)
    offset, matrix = _calibration(study['panel_a'])
    q = np.linalg.solve(matrix.T, (combined - offset).T).T
    panel = study['panel_c']
    _, r_op = _panel_operators(panel)
    p_coef = _ls(_quadratic_design(panel['u']), panel['v1'])
    common, offdiag = _features(_quadratic_design(q) @ p_coef, r_op(q))
    n = len(x)
    y_hat = _ridge(common[:n], train['y'], common[n:], 1e-6)
    return {'common': common, 'offdiag': offdiag, 'y_hat': y_hat}
