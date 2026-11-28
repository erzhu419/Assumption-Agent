import numpy as np
from scipy.linalg import lstsq, solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _operator(u, v, kind):
    def basis(x, q):
        z = x * q
        f = np.arcsinh(z) if kind == 1 else np.logaddexp(z, -z) - np.log(2.)
        return np.column_stack((_aff(x), f))

    def residual(q):
        a = basis(u, q)
        return (a @ lstsq(a, v)[0] - v).ravel()

    q = least_squares(residual, np.full(4, 3.5 if kind == 1 else 1.1),
                      bounds=(0.2, 8.), max_nfev=100).x
    c = lstsq(basis(u, q), v)[0]
    return lambda x: basis(x, q) @ c


def _coordinates(study, x):
    a, b, c = (study[k] for k in ('panel_a', 'panel_b', 'panel_c'))
    i, j = np.triu_indices(len(a['u']), 1)
    offset, scale = [], []
    for u, v in zip(a['u'].T, a['v'].T):
        du, dv = u[i] - u[j], v[i] - v[j]
        ok = abs(du) > 1e-12
        m = np.median(dv[ok] / du[ok]); r = v - m * u
        r = abs(r - np.median(r))
        keep = r <= max(7 * np.median(r), 1e-9)
        b0, m = lstsq(_aff(u[keep, None]), v[keep])[0]
        offset.append(b0); scale.append(m)
    a = (x - offset) / scale
    B = lstsq(_aff(b['u']), b['v'])[0]
    ab = _aff(a) @ B
    c1 = _operator(c['u'], c['v1'], 1)
    c2 = _operator(c['u'], c['v2'], 2)
    odd = c1(ab)
    affine_v2 = _aff(ab) @ lstsq(_aff(c['u']), c['v2'])[0]
    return (np.column_stack((ab, odd, c2(ab))),
            np.column_stack((ab, odd, affine_v2)))


def _basis(t, traj):
    def base(q):
        return -np.expm1(-t[:, None] * np.sort(q))

    def residual(q):
        f = base(q)
        return (f @ lstsq(f, traj.T)[0] - traj.T).ravel()

    fit = least_squares(residual, [.4, 1.8], bounds=(0.03, 15.))
    return base(fit.x)


def _pred(K, I, J, Y, lam):
    a = K[np.ix_(I, I)].copy()
    a.flat[::len(I)+1] += lam
    # Fit an unpenalized constant alongside the smooth component.
    rhs = np.column_stack((Y[I], np.ones(len(I))))
    c = solve(a, rhs, assume_a='pos')
    offset = c[:, :-1].sum(axis=0)/c[:, -1].sum()
    return K[np.ix_(J, I)] @ (c[:, :-1]-c[:, -1, None]*offset) + offset


def _kernel(z, I, h, weight):
    z = (z-z[I].mean(axis=0))/np.maximum(z[I].std(axis=0), 1e-8)
    K = (1 + z@z.T/z.shape[1])**2
    d = [cdist(g, g, 'sqeuclidean')/g.shape[1] for g in np.split(z, 3, axis=1)]
    return K + weight*(sum(np.exp(-h*u) for u in d)/3 + np.exp(-h*sum(d)/3))


def _predict(study, report=False):
    tr = study['train']; x = tr['x']; y = tr['y']; T = tr['traj']
    t = np.asarray(study['eval_times']); n = len(x)
    rows = np.arange(n); test = np.arange(n, n+len(study['test']['x']))
    z = _coordinates(study, np.concatenate((x, study['test']['x'])))
    radius = np.sum(((x-x.mean(0))/x.std(0))**2, axis=1)
    folds = [*np.array_split(np.random.default_rng(498).permutation(n), 4),
             np.argsort(radius)[-max(6, n//7):]]
    splits = []
    for J in folds:
        I = np.setdiff1d(rows, J)
        F = _basis(t, T[I])
        Y = np.zeros((n, 3)); Y[I] = np.column_stack((y[I], lstsq(F, T[I].T)[0].T))
        splits.append((I, J, F, Y))
    best = [(np.inf, None), (np.inf, None)]
    for h in (.035, .1, .25):
        for weight in (.03, .12, .4):
            ks = [tuple(_kernel(q[:n], I, h, weight) for q in z)
                  for I, _, _, _ in splits]
            for lam in (1e-6, 3e-5, .0005, .003, .015):
                e = []
                for (ky, kt), (I, J, F, Y) in zip(ks, splits):
                    py = _pred(ky, I, J, Y[:, :1], lam)[:, 0]
                    pt = _pred(kt, I, J, Y[:, 1:], lam) @ F.T
                    d = np.abs(pt - T[J])
                    e.append((np.abs(py-y[J]).mean(), .4*d.mean()+.6*d[:, -1].mean()))
                e = np.asarray(e)
                score = .50*e[:-1].mean(0)+.15*e[:-1].max(0)+.35*e[-1]
                for j, val in enumerate(score):
                    if val < best[j][0]:
                        best[j] = val, (h, weight, lam)
    if report:
        print('inner selected', best)
    F = _basis(t, T); Y = np.column_stack((y, lstsq(F, T.T)[0].T))
    predictions = [_pred(_kernel(q, rows, h, w), rows, test, Y, lam)
                   for q, (_, (h, w, lam)) in zip(z, best)]
    return predictions[0][:, 0], predictions[1][:, 1:] @ F.T


def solve_batch(study: dict) -> dict:
    y, state = _predict(study)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
