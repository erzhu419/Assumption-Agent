import numpy as np


def _fit_panel_a(panel):
    u, v = panel['u'], panel['v']
    affine = np.column_stack((np.ones(len(u)), u))
    designs = np.stack([np.column_stack((affine, u[:, j] * u[:, (j + 1) % 4], u[:, j] ** 3)) for j in range(4)])
    weights = np.ones(len(u))
    for _ in range(25):
        root = np.sqrt(weights)
        coefs = np.column_stack([np.linalg.lstsq(designs[j] * root[:, None], v[:, j] * root, rcond=None)[0] for j in range(4)])
        errors = np.linalg.norm(np.einsum('jni,ij->nj', designs, coefs) - v, axis=1)
        scale = max(2 * np.median(errors), 1e-4)
        weights = 1 / (1 + (errors / scale) ** 2)
    return coefs


def _inverse_a(x, coef):
    eye = np.eye(4)
    adjacent = np.roll(eye, 1, axis=1)
    z = np.linalg.solve(coef[1:5].T, (x - coef[0]).T).T
    for _ in range(20):
        nxt = np.roll(z, -1, axis=1)
        res = coef[0] + z @ coef[1:5] + coef[5] * z * nxt + coef[6] * z ** 3 - x
        if np.max(np.abs(res)) < 1e-11:
            break
        jac = coef[1:5].T + (coef[5] * nxt + 3 * coef[6] * z ** 2)[:, :, None] * eye + (coef[5] * z)[:, :, None] * adjacent
        z -= np.linalg.solve(jac, res[..., None])[..., 0]
    return z


def _fit_panel_b(panel):
    u = panel['u']
    return np.linalg.lstsq(np.column_stack((np.ones(len(u)), u)), panel['v'], rcond=None)[0]


def _coordinate_model(q, v, y, traj):
    from scipy.optimize import least_squares

    i, j = np.triu_indices(4)
    Q = q[:, i] * q[:, j]
    X = np.column_stack((np.ones(len(q)), q, Q))
    co = np.linalg.lstsq(X, traj, rcond=None)[0]
    U, _, _ = np.linalg.svd(co[5:], full_matrices=False)
    carrier = U[:, 0] / U[1, 0]
    keep = np.arange(10) != 1
    g = Q @ carrier
    one = np.ones(len(q))
    y = np.asarray(y).reshape(-1)
    L = np.linalg.lstsq(np.column_stack((one, v, g)), traj, rcond=None)[0]
    lin = L[1:5]
    start = np.array([0., 1., 1., 1., -1., 0., 0.])
    rng = np.random.default_rng(130)

    def unpack(p):
        c = carrier.copy()
        c[keep] += p[11:20]
        return c, np.eye(4) + p[20:].reshape(4, 4)

    def residual(p):
        c, mix = unpack(p)
        z = v @ mix + (Q @ c)[:, None] * p[:4]
        A = np.column_stack((one, z))
        F = np.column_stack((A, z[:, 0]*z[:, 2], z[:, 1]*z[:, 3]))
        D = np.column_stack((F, np.tanh(F @ p[4:11])))
        ry = D @ np.linalg.lstsq(D, y, rcond=None)[0] - y
        rt = A @ np.linalg.lstsq(A, traj, rcond=None)[0] - traj
        return np.r_[ry, rt.ravel(), .004*p[:11], .35*p[11:20], .05*p[20:]]

    starts = [np.linalg.solve(lin@lin.T+r*r*np.eye(4), lin@L[5]) for r in (.03, .1, .3)]
    starts += [np.eye(4)[k] * (lin[k] @ L[5])/(lin[k] @ lin[k]) for k in range(4)]
    best = None
    for k, d in enumerate(starts):
        for m in range(3):
            theta = start if m == 0 else rng.normal(0, .8, 7)
            p0 = np.r_[d, theta, np.zeros(25)]
            sol = least_squares(residual, p0, max_nfev=160, ftol=2e-6)
            if best is None or sol.cost < best.cost:
                best = sol
    c, mix = unpack(best.x)
    return c, mix, best.x[:4]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    a = _fit_panel_a(study['panel_a'])
    b = _fit_panel_b(study['panel_b'])
    q = _inverse_a(train['x'], a)
    qt = _inverse_a(test['x'], a)
    v = q @ b[1:] + b[0]
    c, mix, d = _coordinate_model(q, v, train['y'], train['traj'])
    i, j = np.triu_indices(4)
    def transform(x):
        return (x @ b[1:] + b[0]) @ mix + ((x[:, i]*x[:, j]) @ c)[:, None]*d
    return {'z': transform(q), 'z_test': transform(qt), 'y': train['y'], 'traj': train['traj']}
