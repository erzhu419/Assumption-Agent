import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _operator(u, v, transform, limits, mask=None):
    if mask is not None:
        u, v = u[mask], v[mask]

    def design(x, k):
        return np.column_stack((_affine(x), transform(x, k)))

    def objective(k):
        a = design(u, k)
        return np.sum((a @ _fit(a, v) - v)**2)

    grid = np.linspace(*limits, 41)
    i = int(np.argmin([objective(k) for k in grid]))
    k = minimize_scalar(objective, bounds=(grid[max(i-1, 0)], grid[min(i+1, 40)]),
                        method='bounded', options={'xatol': 1e-9}).x
    w = _fit(design(u, k), v)
    return lambda x: design(x, k) @ w


def _coordinates(panel, x):
    u, v = panel['u'], panel['v']
    d = np.linalg.norm(v-u, axis=1)
    mask = d < max(4., 4.*np.median(d))
    op = _operator(u, v, lambda q,k: np.sin(k*q), (1.5, 3.), mask)
    a = _fit(_affine(u[mask]), v[mask])
    seeds = np.linalg.solve(a[1:].T, (x-a[0]).T).T
    result = []
    for row, seed in zip(x, seeds):
        def residual(q):
            return op(q[None, :])[0] - row
        fit = root(residual, seed, tol=1e-10)
        if np.linalg.norm(residual(fit.x)) > 1e-8:
            fit = least_squares(residual, seed, max_nfev=100)
        result.append(fit.x)
    return np.asarray(result).reshape(x.shape)


def _operators(panel, q):
    u = panel['u']
    p = _operator(u, panel['v1'], lambda x,k: x/(1+k*abs(x)), (.2, 2.))(q)
    r = _operator(u, panel['v2'], lambda x,k: x*x/(1+k*x*x), (.02, .6))(q)
    return p, r


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    q = _coordinates(study['panel_a'], x)
    p, r = _operators(study['panel_c'], q)
    joint = p[:, :, None]*r[:, None, :]
    trace = np.trace(joint, axis1=1, axis2=2)
    fy = _affine(np.column_stack((p, r, trace)))
    z = train['traj']
    basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    target = z @ basis.T

    # The shared trace captures the diagonal interactions. Select one
    # off-diagonal interaction using only the fitted trajectory rows.
    terms = joint.reshape(len(q), 16)[:, [i for i in range(16) if i % 5]]
    residual = target - fy[:n] @ _fit(fy[:n], target)
    perpendicular = terms[:n] - fy[:n] @ _fit(fy[:n], terms[:n])
    energy = np.sum(perpendicular**2, axis=0)
    gains = np.sum((perpendicular.T @ residual)**2, axis=1) / np.maximum(energy, 1e-20)
    fs = np.column_stack((fy, terms[:, np.argmax(gains)]))
    y = fy[n:] @ _fit(fy[:n], train['y'])
    state = fs[n:] @ _fit(fs[:n], target) @ basis
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
