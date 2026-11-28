import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ls(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-10)[0]


def _next(x):
    return np.roll(x, -1, axis=1)


def _calibration_design(x):
    z = _next(x)
    return np.stack((np.ones_like(x), x, z, x*z, x**3), axis=-1)


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    f = _calibration_design(u)
    mask = np.ones(len(u), dtype=bool)
    for _ in range(12):
        w = np.array([_ls(f[mask, j], v[mask, j]) for j in range(u.shape[1])])
        e = np.linalg.norm(np.einsum('nij,ij->ni', f, w) - v, axis=1)
        new = e <= max(1e-9, 3.5*np.median(e))
        if new.sum() < 2*f.shape[-1] or np.array_equal(mask, new):
            break
        mask = new
    w = np.array([_ls(f[mask, j], v[mask, j]) for j in range(u.shape[1])])
    j = np.arange(u.shape[1])
    neighbor = np.roll(j, -1)

    def jac(q):
        J = np.zeros((len(q), len(q)))
        J[j, j] = w[:, 1] + w[:, 3]*q[neighbor] + 3*w[:, 4]*q*q
        J[j, neighbor] = w[:, 2] + w[:, 3]*q
        return J

    def forward(q):
        return np.sum(_calibration_design(q[None])[0]*w, axis=1)

    a = jac(np.zeros(u.shape[1]))
    initial = np.linalg.solve(a, (x-w[:, 0]).T).T
    return np.array([least_squares(lambda q: forward(q)-obs, guess, jac=jac,
                                   gtol=1e-11, max_nfev=80).x
                     for obs, guess in zip(x, initial)])


def _operator_design(x, k, coupled):
    # For k>0 this is the span of x and x/(1+k*abs(x)); the
    # reparameterization also gives a well-conditioned basis at k=0.
    h = x*np.abs(x)/(1+k*np.abs(x))
    columns = (np.ones(len(x)), x, h)
    if coupled:
        columns += (x*_next(x),)
    return np.column_stack(columns)


def _operator(u, v, q, coupled):
    def loss(k):
        a = _operator_design(u, k, coupled)
        return np.linalg.norm(a @ _ls(a, v) - v)

    opt = minimize_scalar(loss, bounds=(0., 7.), method='bounded')
    k = min((0., float(opt.x), 7.), key=loss)
    w = _ls(_operator_design(u, k, coupled), v)
    return _operator_design(q, k, coupled) @ w


def _predict(f, y, query):
    fm, ym = f.mean(axis=0), y.mean(axis=0)
    a = f-fm
    w = np.linalg.solve(a.T @ a + 1e-6*np.eye(a.shape[1]), a.T @ (y-ym))
    return (query-fm) @ w + ym


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['x'])
    q = _calibrate(study['panel_a'], np.concatenate((train['x'], study['test']['x'])))
    panel = study['panel_c']
    p = _operator(panel['u'], panel['v1'], q, False)
    r = _operator(panel['u'], panel['v2'], q, True)
    common = np.column_stack((p, r, np.sum(p*r, axis=1)))
    offdiag = np.column_stack([p[:, i]*r[:, j] for i in range(p.shape[1])
                               for j in range(r.shape[1]) if i != j])
    return {'common': common, 'offdiag': offdiag,
            'y_hat': _predict(common[:n], train['y'], common[n:])}
