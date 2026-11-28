import numpy as np
from scipy.optimize import least_squares, root

def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]

def one(z):
    return np.c_[np.ones(len(z)), z]

def _panel_maps(panel, z):
    u = np.asarray(panel['u'], dtype=float)

    # Both responses share their nonlinear shape parameters across coordinates.
    def softsign(w, q):
        return one(w / (1 + q[0]*abs(w)) + q[1]*w)

    def curvature(w, q):
        w2 = w*w
        return np.c_[one(w), w2, w2/(1+q[0]*w2)]

    def fit(features, target, initial, bounds, **options):
        def residual(q):
            f = features(u, q)
            return (f @ lin(f, target) - target).ravel()
        q = least_squares(residual, initial, bounds=bounds, **options).x
        return features(z, q) @ lin(features(u, q), target)

    c = fit(softsign, panel['v1'], [.85, -.01], ([.02, -1], [10, 1]))
    d = fit(curvature, panel['v2'], [.25], (.08, .65), gtol=1e-13)
    return np.c_[c, d]


def _invert_panel(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    affine = one(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(4):
        coef = lin(affine[keep], v[keep])
        keep = np.linalg.norm(v-affine@coef, axis=1) < 2
    nxt = np.roll(u, -1, axis=1)
    coef = np.column_stack([
        lin(np.column_stack((np.ones(len(u)), u[:, j], nxt[:, j],
                            u[:, j]*nxt[:, j], u[:, j]**3))[keep], v[keep, j])
        for j in range(u.shape[1])
    ])
    b, scale, mix, cross, cubic = coef
    aff = lin(affine[keep], v[keep])
    def forward(z):
        nz = np.roll(z, -1)
        return b + scale*z + mix*nz + cross*z*nz + cubic*z**3
    def jac(z):
        nz = np.roll(z, -1)
        return np.diag(scale + cross*nz + 3*cubic*z*z) + np.diag(mix+cross*z)@np.roll(np.eye(len(z)), -1, axis=0)
    init = np.linalg.solve(aff[1:].T, (x-aff[0]).T).T
    out = []
    for row, guess in zip(x, init):
        r = root(lambda z:forward(z)-row, guess, jac=jac, tol=1e-10)
        out.append(r.x if np.linalg.norm(r.fun) < 1e-8 else least_squares(lambda z:forward(z)-row, r.x, jac=jac).x)
    return np.asarray(out)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train']['x'], study['test']['x']
    n = len(train)
    z = _invert_panel(study['panel_a'], np.concatenate((train, test)))
    panel = study['panel_c']
    u, v = panel['u'], panel['v1']
    def design(x, q):
        return one(np.column_stack((x, np.tanh(q[0]*x), x*np.abs(x))))
    def residual(q):
        f = design(u, q)
        return (f@lin(f, v)-v).ravel()
    q = min((least_squares(residual, [s], bounds=(0.02, 10), max_nfev=300)
             for s in (0.25, 0.85, 1.6, 2.5)), key=lambda fit:fit.cost).x
    c = design(z, q)@lin(design(u, q), v)
    d = _panel_maps(panel, z)[:, u.shape[1]:]
    w = np.column_stack((c, d))
    w = (w-w[:n].mean(axis=0))/np.maximum(w[:n].std(axis=0), 0.1)
    return {'features': w, 'n_train': np.array(n)}
