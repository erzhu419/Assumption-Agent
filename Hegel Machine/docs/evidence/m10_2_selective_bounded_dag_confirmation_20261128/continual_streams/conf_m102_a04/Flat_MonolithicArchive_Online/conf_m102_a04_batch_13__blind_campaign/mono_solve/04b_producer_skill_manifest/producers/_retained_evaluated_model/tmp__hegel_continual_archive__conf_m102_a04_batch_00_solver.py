import numpy as np
from scipy.optimize import least_squares


def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def one(z):
    return np.c_[np.ones(len(z)), z]


def _align(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    offset, scale = [], []
    for a, b in zip(u.T, v.T):
        F = one(a)
        ij = np.triu_indices(len(a), 1)
        da = (a[:, None] - a)[ij]
        db = (b[:, None] - b)[ij]
        mask = abs(da) > 1e-8
        slope = np.median(db[mask] / da[mask])
        inter = np.median(b - slope*a)
        for _ in range(5):
            r = abs(b - inter - slope*a)
            keep = r <= max(5*np.median(r), 0.00035)
            inter, slope = lin(F[keep], b[keep])
        offset.append(inter)
        scale.append(slope)
    return np.asarray(offset), np.asarray(scale)


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


def _time_basis(t, trajectory):
    def basis(r):
        return -np.expm1(-t[:, None]*r)

    def residual(r):
        f = basis(r)
        return (f @ lin(f, trajectory.T) - trajectory.T).ravel()

    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .8], [.8, 4])).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    offset, scale = _align(study['panel_a'])
    z = (np.vstack((train['x'], study['test']['x'])) - offset)/scale
    w = _panel_maps(study['panel_c'], z)
    w = (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1)
    gram = w @ w.T / w.shape[1]
    norms = gram.diagonal()
    distance = np.maximum(norms[:, None] + norms - 2*gram, 0)
    quadratic = (1+.25*gram)**2
    targets = np.column_stack((train['y'], train['traj']))
    eye = np.eye(n)
    fit = np.mean([
        kernel[n:, :n] @ np.linalg.solve(kernel[:n, :n] + ridge*eye, targets)
        for kernel, ridge in ((quadratic, 3e-5),
                              (quadratic + np.exp(-.06*distance), 3e-4))
    ], axis=0)
    y = fit[:, 0]
    basis = _time_basis(np.asarray(study['eval_times']), train['traj'])
    state = (basis @ lin(basis, fit[:, 1:].T)).T
    terminal = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
