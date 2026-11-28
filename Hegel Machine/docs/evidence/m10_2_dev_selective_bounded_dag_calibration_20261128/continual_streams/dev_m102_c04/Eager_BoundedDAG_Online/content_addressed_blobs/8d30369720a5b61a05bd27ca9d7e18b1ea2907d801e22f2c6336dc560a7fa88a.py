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


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    q = _invert_calibration(study["panel_a"], x)
    p_op, r_op = _panel_operators(study['panel_c'])
    common, offdiag = _features(p_op(q), r_op(q))
    n = len(train['x'])
    y = _affine(common[n:]) @ _ls(_affine(common[:n]), train['y'])
    return {'common': common, 'offdiag': offdiag, 'y_hat': y}


def _invert_calibration(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    b0, w0 = _calibration(panel)
    e = np.linalg.norm(u @ w0 + b0 - v, axis=1)
    keep = e <= max(1e-9, 4*np.median(e))
    if keep.sum() < 12:
        return np.linalg.solve(w0.T, (x-b0).T).T
    u, v = u[keep], v[keep]
    pair = np.array([2, 3, 0, 1])

    def design(z, rates, j):
        a, c = z[:, j], z[:, pair[j]]
        return np.column_stack((np.ones(len(z)), a, c, np.sin(rates[0]*a), np.sin(rates[1]*c)))

    def fit(rates, ret=False):
        mats = [design(u, rates, j) for j in range(4)]
        w = np.array([_ls(a, v[:, j]) for j, a in enumerate(mats)])
        return w if ret else np.column_stack([a@w[j]-v[:, j] for j, a in enumerate(mats)]).ravel()

    opt = least_squares(fit, [2.2, 2.4], bounds=([.5,.5],[4.5,4.5]), max_nfev=75)
    rates = opt.x
    w = fit(rates, True)
    if np.sqrt(np.mean(opt.fun**2)) > .65*np.sqrt(np.mean((u @ w0 + b0 - v)**2)):
        return np.linalg.solve(w0.T, (x-b0).T).T
    q = np.linalg.solve(w0.T, (x-b0).T).T
    def forward(z):
        return np.column_stack([design(z, rates, j)@w[j] for j in range(4)])
    for _ in range(24):
        r = forward(q)-x
        if np.max(np.abs(r)) < 1e-11:
            break
        jac = np.zeros((len(x), 4, 4))
        j = np.arange(4)
        jac[:, j, j] = w[:, 1]+w[:, 3]*rates[0]*np.cos(rates[0]*q)
        jac[:, j, pair] = w[:, 2]+w[:, 4]*rates[1]*np.cos(rates[1]*q[:,pair])
        step = np.linalg.solve(jac, r[...,None])[...,0]
        old = np.sum(r*r, axis=1)
        next_q = q.copy()
        pending = np.ones(len(q), dtype=bool)
        for k in range(12):
            trial = q - step * (0.5**k)
            take = pending & (np.sum((forward(trial)-x)**2, axis=1) < old)
            next_q[take] = trial[take]
            pending[take] = False
            if not pending.any():
                break
        q = next_q
    return q
