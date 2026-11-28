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
    abi = solve(B[1:].T, (a - B[0]).T).T
    c1 = _operator(c['u'], c['v1'], 1)
    c2 = _operator(c['u'], c['v2'], 2)
    return dict(raw=x, a=a, ab=ab, abi=abi, ac=c1(a), abc=c1(ab),
                abd=np.column_stack((ab, c2(ab))), abcd=np.column_stack((ab, c1(ab), c2(ab))))


def _basis(t, traj):
    def base(q):
        return -np.expm1(-t[:, None] * np.sort(q))

    def residual(q):
        f = base(q)
        return (f @ lstsq(f, traj.T)[0] - traj.T).ravel()

    fit = least_squares(residual, [.4, 1.8], bounds=(0.03, 15.))
    return base(fit.x)


def _kernels(z):
    d = cdist(z, z, 'sqeuclidean') / z.shape[1]
    r = np.sqrt(d)
    for h in (.55, 1., 1.7):
        q = h*r
        yield 'm'+str(h), (1+q+q*q/3)*np.exp(-q)
    for h in (.2, .5, 1.):
        yield 'r'+str(h), np.exp(-h*d)
    if z.shape[1] == 4:
        dz = (z[:, None, :] - z[None, :, :])**2
        for h in (.3, .7):
            u = np.exp(-h*dz)
            v = u.mean(axis=2)
            w = (u.sum(axis=2)**2-(u*u).sum(axis=2))/12
            yield 'anova'+str(h), .20*v+.40*w+.40*np.exp(-h*d)


def _pred(K, I, J, Y, lam):
    a = K[np.ix_(I, I)].copy()
    a.flat[::len(I)+1] += lam
    # Fit an unpenalized constant alongside the smooth component.
    rhs = np.column_stack((Y[I], np.ones(len(I))))
    c = solve(a, rhs, assume_a='pos')
    offset = c[:, :-1].sum(axis=0)/c[:, -1].sum()
    return K[np.ix_(J, I)] @ (c[:, :-1]-c[:, -1, None]*offset) + offset


def _predict(study, report=False):
    tr = study['train']; x = tr['x']; y = tr['y']; T = tr['traj']
    F = _basis(np.asarray(study['eval_times']), T)
    Y = np.column_stack((y, lstsq(F, T.T)[0].T))
    n = len(x); rows = np.arange(n); test = np.arange(n, n+len(study['test']['x']))
    coords = _coordinates(study, np.concatenate((x, study['test']['x'])))
    folds = list(np.array_split(np.random.default_rng(498).permutation(n), 4))
    # Add a high-radius slice to penalize fits that only interpolate the center.
    z = coords['a'][:n]; radius = np.sum(((z-z.mean(0))/z.std(0))**2, axis=1)
    folds.append(np.argsort(radius)[-max(6, n//7):])
    splits = [(np.setdiff1d(rows, J), J) for J in folds]
    records = []
    for key, z in coords.items():
        z = (z-z[:n].mean(0))/np.maximum(z[:n].std(0), 1e-8)
        for kind, K in _kernels(z):
            for lam in (1e-6, 1e-4, .001, .006, .025):
                e = []
                for I, J in splits:
                    p = _pred(K, I, J, Y, lam)
                    er = np.abs(p[:, 1:] @ F.T - T[J])
                    e.append([np.mean(abs(p[:, 0]-y[J])), .4*er.mean()+.6*er[:, -1].mean()])
                e = np.array(e)
                score = .55*e[:-1].mean(0)+.20*e[:-1].max(0)+.25*e[-1]
                records.append((score, key, kind, lam, e, K))
    result = []
    for j in range(2):
        best = sorted(records, key=lambda r: r[0][j])
        if report:
            print('inner', j, [(k, f, l, np.round(e[:, j], 4).tolist()) for _, k, f, l, e, _ in best[:3]])
        # Average near-ties across different coordinate maps instead of locking in a noisy ranking.
        ensemble = []
        used = set(); weights = []
        threshold = best[0][0][j]
        for score, key, kind, lam, _, K in best:
            if key in used:
                continue
            if score[j] > 1.17*threshold or len(used) == 4:
                break
            used.add(key)
            weights.append(np.exp(-9*(score[j]/max(threshold, 1e-12)-1)))
            ensemble.append(_pred(K, rows, test, Y, lam))
        p = np.average(ensemble, axis=0, weights=weights)
        result.append(p[:, 0] if j == 0 else p[:, 1:] @ F.T)
    return result


def solve_batch(study: dict) -> dict:
    y, state = _predict(study)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
