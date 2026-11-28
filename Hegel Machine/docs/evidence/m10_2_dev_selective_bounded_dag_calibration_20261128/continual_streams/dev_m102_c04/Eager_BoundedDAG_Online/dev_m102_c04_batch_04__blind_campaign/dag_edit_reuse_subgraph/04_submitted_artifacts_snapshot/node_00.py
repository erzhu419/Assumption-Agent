import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    a = _affine(u)
    mask = np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = _ls(a[mask], v[mask])
        e = np.linalg.norm(a @ w - v, axis=1)
        new = e <= max(1e-10, 4 * np.median(e))
        if new.sum() < 8 or np.array_equal(mask, new):
            break
        mask = new
    d = np.array([_ls(a[mask][:, [0, j+1]], v[mask, j]) for j in range(4)]).T
    if np.median(np.linalg.norm((u*d[1]+d[0]-v)[mask], axis=1)) <= 1.35*np.median(e[mask])+1e-10:
        return d[0], np.diag(d[1])
    return w[0], w[1:]


def _panel_operators(panel):
    u = panel['u']

    def bilinear(x):
        return np.column_stack((np.ones(len(x)), x[:, 0]*x[:, 1],
                                x[:, 0]*x[:, 3], x[:, 1]*x[:, 2],
                                x[:, 2]*x[:, 3]))

    wp = _ls(bilinear(u), panel['v1'])

    def saturating(x, k):
        sq = x*x
        return np.column_stack((_affine(x), sq/(1+k*sq)))

    v = panel['v2']
    def residual(k):
        f = saturating(u, k[0])
        return (f @ _ls(f, v) - v).ravel()

    k = least_squares(residual, [0.16], bounds=(0.01, 5.)).x[0]
    wr = _ls(saturating(u, k), v)
    return lambda x: bilinear(x) @ wp, lambda x: saturating(x, k) @ wr


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


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    offset, matrix = _calibration(study['panel_a'])
    q = np.linalg.solve(matrix.T, (x-offset).T).T
    p_op, r_op = _panel_operators(study['panel_c'])
    common, offdiag = _features(p_op(q), r_op(q))
    n = len(train['x'])
    y = _ridge(common[:n], train['y'], common[n:], 1e-6)
    return {'common': common, 'offdiag': offdiag, 'y_hat': y}
