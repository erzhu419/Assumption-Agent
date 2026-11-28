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


def _select_interaction(common, offdiag, target):
    # Compare each candidate after removing the contribution of the shared basis.
    a = _affine(common)
    residual = target - a @ _ls(a, target)
    orthogonal = offdiag - a @ _ls(a, offdiag)
    energy = np.sum(orthogonal**2, axis=0)
    gain = np.sum((orthogonal.T @ residual)**2, axis=1) / np.maximum(energy, 1e-20)
    best = int(np.argmax(gain))
    return best if gain[best] > 0.15*np.sum(residual**2) else None


def _temporal_basis(times, trajectories):
    t = np.asarray(times)
    def basis(rates):
        return -np.expm1(-t[:, None]*rates)
    def err(rates):
        f = basis(rates)
        return (f @ _ls(f, trajectories.T) - trajectories.T).ravel()
    opt = least_squares(err, [1.8, .43], bounds=([.8, .03], [8., .79]), max_nfev=80)
    return np.linalg.qr(basis(opt.x), mode='reduced')[0]


def _ridge(f, target, query, penalty):
    fm, ym = f.mean(axis=0), target.mean(axis=0)
    a = f-fm
    w = np.linalg.solve(a.T @ a + penalty*np.eye(f.shape[1]), a.T @ (target-ym))
    return (query-fm) @ w + ym


def solve_batch(study):
    train = study['train']
    x, xt = train['x'], study['test']['x']
    offset, matrix = _calibration(study['panel_a'])
    q = np.linalg.solve(matrix.T, (np.concatenate((x, xt))-offset).T).T
    p_op, r_op = _panel_operators(study['panel_c'])
    common, offdiag = _features(p_op(q), r_op(q))
    n = len(x)
    basis = _temporal_basis(study['eval_times'], train['traj'])
    scores = train['traj'] @ basis
    y = _ridge(common[:n], train['y'], common[n:], 1e-6)
    j = _select_interaction(common[:n], offdiag[:n], scores)
    f = common if j is None else np.column_stack((common, offdiag[:, j]))
    state = _ridge(f[:n], scores, f[n:], 1e-6) @ basis.T
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
