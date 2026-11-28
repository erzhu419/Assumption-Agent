"""Recover calibrated latent channels and fit two relaxation modes."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-9)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']

    def features(z):
        w = np.roll(z, -1, axis=1)
        return np.stack((np.ones_like(z), z, w, z*w, z**3), axis=-1)

    a = features(u)
    rng = np.random.default_rng(74)
    best_score = np.inf
    for _ in range(96):
        g = rng.choice(len(u), 8, replace=False)
        c = np.array([_ls(a[g, j], v[g, j]) for j in range(4)])
        error = np.max(np.abs(np.einsum('ijk,jk->ij', a, c)-v), axis=1)
        score = np.median(error)
        if score < best_score:
            best_score, best_error = score, error
    good = best_error < max(0.001, 6*best_score)
    bias, diag, adj, pair, cube = np.array([_ls(a[good, j], v[good, j]) for j in range(4)]).T
    j = np.arange(4)
    jp = (j+1) % 4
    jac = np.zeros((len(x), 4, 4))
    linear = np.diag(diag)
    linear[j, jp] = adj
    z = np.linalg.solve(linear, (x-bias).T).T
    for _ in range(30):
        w = z[:, jp]
        r = bias+diag*z+adj*w+pair*z*w+cube*z**3-x
        if np.max(np.abs(r), initial=0) < 1e-11:
            break
        jac[:, j, j] = diag+pair*w+3*cube*z*z
        jac[:, j, jp] = adj+pair*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step / np.maximum(1, np.linalg.norm(step, axis=1, keepdims=True))
    return z


def _branches(panel, z):
    def left(x, k):
        v = k*x
        return _design(np.column_stack((x, v/(1+np.abs(v)))))

    def right(x, k):
        return _design(np.column_stack((x, x*np.roll(x, -1, axis=1), np.arctan(k*x))))

    u = panel['u']
    out = []
    for key, design, bounds in (('v1', left, (0.4, 1.5)), ('v2', right, (0.65, 1.55))):
        v = panel[key]

        def loss(k):
            a = design(u, k)
            q = np.linalg.qr(a, mode='reduced')[0]
            e = (q @ (q.T @ v)-v)/(1-np.sum(q*q, axis=1, keepdims=True))
            return np.mean(e*e)

        k = minimize_scalar(loss, bounds=bounds, method='bounded').x
        out.append(design(z, k) @ _ls(design(u, k), v))
    return out


def _predict_trajectory(times, observed, fast, slow):
    n = len(observed)
    target = observed.ravel()

    def basis(rates):
        return -np.expm1(-times[:, None]*rates)

    def design(rates, rows):
        h = basis(rates)
        shape = (len(fast[rows])*len(times), fast.shape[1]+slow.shape[1])
        return np.concatenate((fast[rows, None, :]*h[None, :, :1],
                               slow[rows, None, :]*h[None, :, 1:]), axis=-1).reshape(shape)

    train, test = slice(n), slice(n, None)

    def residual(rates):
        a = design(rates, train)
        return a @ _ls(a, target) - target

    fit = least_squares(residual, [1.83, 0.42], bounds=([1.1, 0.1], [3., 0.9]))
    return (design(fit.x, test) @ _ls(design(fit.x, train), target)).reshape(len(fast)-n, len(times))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    z = _calibrate(study['panel_a'], np.concatenate((train['x'], study['test']['x'])))
    p, q = _branches(study['panel_c'], z)
    dot = np.sum(p*q, axis=1)
    cross = p[:, 0]*q[:, 1]
    scalar = _design(np.column_stack((p, q, dot)))
    fast = np.column_stack((scalar, cross))
    slow = _design(np.column_stack((q, cross)))
    y = scalar[n:] @ _ls(scalar[:n], train['y'])
    traj = _predict_trajectory(np.asarray(study['eval_times']), train['traj'], fast, slow)
    last = traj[:, -1]
    return dict(y_hat=y, state_pred=traj, diagnostics_vec=np.array([y.mean(), y.std(), last.mean(), last.std()]))
