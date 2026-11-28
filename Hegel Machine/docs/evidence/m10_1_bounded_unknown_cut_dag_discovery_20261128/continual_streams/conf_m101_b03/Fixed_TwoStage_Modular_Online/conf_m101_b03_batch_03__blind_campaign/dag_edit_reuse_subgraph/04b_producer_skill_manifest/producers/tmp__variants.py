import numpy as np


# Panel A couples each coordinate to its cyclic successor.
_NEXT = np.array([1, 2, 3, 0])
_IDX = np.arange(4)


def _fit_panel_a(u, v):
    affine = np.column_stack((np.ones(len(u)), u))
    pair = u * u[:, _NEXT]
    cube = u ** 3
    designs = [np.column_stack((affine, pair[:, j], cube[:, j])) for j in _IDX]
    weight = np.ones(len(u))

    def fit():
        return np.column_stack([
            np.linalg.lstsq(d * weight[:, None], v[:, j] * weight, rcond=None)[0]
            for j, d in enumerate(designs)
        ])

    for _ in range(25):
        coef = fit()
        predicted = np.column_stack([d @ coef[:, j] for j, d in enumerate(designs)])
        residual = np.linalg.norm(predicted - v, axis=1)
        scale = max(np.median(residual), 1e-8)
        weight = 1 / np.hypot(1, residual / scale)
    weight[np.argsort(residual)[-3:]] = 0
    return fit()


def _a_forward(u, coef):
    return coef[0] + u @ coef[1:5] + coef[5] * u * u[:, _NEXT] + coef[6] * u ** 3


def _panel_a(panel, observed):
    coef = _fit_panel_a(panel['u'], panel['v'])
    u = np.linalg.solve(coef[1:5].T, (observed - coef[0]).T).T
    for _ in range(30):
        residual = _a_forward(u, coef) - observed
        error = np.linalg.norm(residual, axis=1)
        if np.max(error) < 1e-11:
            break
        jac = np.broadcast_to(coef[1:5].T, (len(u), 4, 4)).copy()
        jac[:, _IDX, _IDX] += coef[5] * u[:, _NEXT] + 3 * coef[6] * u ** 2
        jac[:, _IDX, _NEXT] += coef[5] * u
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        trial = u - step
        for _ in range(12):
            worse = np.linalg.norm(_a_forward(trial, coef) - observed, axis=1) > error
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = u[worse] - step[worse]
        u = trial
    return u


def _panel_c_v2(panel, calibrated):
    def design(u):
        return np.column_stack((np.ones(len(u)), np.cos(2.4 * u)))

    coef = np.linalg.lstsq(design(panel['u']), panel['v2'], rcond=None)[0]
    return design(calibrated) @ coef


def _base_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    calibrated = _panel_a(study['panel_a'], np.vstack((train['x'], test['x'])))
    return {
        'z': np.column_stack((calibrated, _panel_c_v2(study['panel_c'], calibrated))),
        'n_train': np.asarray(len(train['x'])),
        'eval_times': np.asarray(study['eval_times']),
        'y': np.asarray(train['y']),
        'traj': np.asarray(train['traj']),
    }
from itertools import combinations, combinations_with_replacement


def _mapped(panel, u, key):
    r = panel['u']
    d = np.column_stack((np.ones(len(r)), r))
    return np.column_stack((np.ones(len(u)), u)) @ np.linalg.lstsq(d, panel[key], rcond=None)[0]


def _feature_groups(z, b, c, pure):
    yield z
    yield b
    yield c
    yield pure
    for a in z.T:
        yield np.column_stack((a, a**2, a**3, a**4))
        yield np.column_stack([f(k*a) for f in (np.sin, np.cos) for k in (1.2, 2.4, 3.6)])
    for q in (z, b, c, pure):
        for i, j in combinations(range(4), 2):
            a, d = q[:, i], q[:, j]
            w = a*d
            yield np.column_stack((w, w*a, w*d, w*w))
            if q is z or q is b:
                yield np.column_stack([f(k*w) for f in (np.sin, np.cos, np.tanh) for k in (1.2, 2.4, 3.6)] + [1/(1+k*w*w) for k in (1., 3., 6.)])
                for x in (a+d, a-d):
                    yield np.column_stack([f(k*x) for f in (np.sin, np.cos) for k in (.8, 1.6, 2.4, 3.2)])
                for x, v in ((a,d), (d,a)):
                    yield np.column_stack((x*v**2, x*v**3, x*np.sin(2.4*v), x*np.cos(2.4*v), x/(1+2*v*v)))
    for a in c.T:
        yield np.column_stack((a*a, a**3, np.sin(1.5*a), np.cos(1.5*a), np.sin(3*a), np.cos(3*a)))
        yield a[:,None]*z
    yield np.column_stack([np.prod(z[:, ids], axis=1) for degree in (3, 4) for ids in combinations_with_replacement(range(4), degree) if len(set(ids)) >= 3])


def _profile_features(z, b, c, pure, y, traj, times):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import least_squares, minimize

    def modes(r):
        h = -np.expm1(-times[:,None]*r)
        return h, np.linalg.lstsq(h, traj.T, rcond=None)[0]

    def residual(r):
        h,m = modes(r)
        return (h@m-traj.T).ravel()

    r = least_squares(residual, [.5,2.], bounds=(.01,10.)).x
    _,m = modes(np.sort(r))
    targets = np.column_stack((y,m.T))
    n = len(y)
    design = []
    for g in _feature_groups(z,b,c,pure):
        a = (g-g[:n].mean(0))/np.maximum(g[:n].std(0),1e-9)
        design.append(a/np.sqrt(a.shape[1]))
    K = np.asarray([a[:n]@a[:n].T for a in design])
    K = np.ascontiguousarray(np.concatenate((K,np.eye(n)[None])).reshape(len(K)+1,-1))
    prior = np.full(len(K), -4.)
    prior[:4] = -2.5
    prior[-1] = -6.
    columns = []
    E = np.eye(n)
    for v in targets.T:
        v = (v-v.mean())/max(v.std(),1e-9)
        def objective(h):
            w = np.exp(h)
            cf = cho_factor((w@K).reshape(n,n))
            a = cho_solve(cf,v)
            g = cho_solve(cf,E)-np.outer(a,a)
            delta = h-prior
            return .5*v@a+np.log(np.diag(cf[0])).sum()+.20*(delta@delta), .5*w*(K@g.ravel())+.40*delta
        h = minimize(objective, prior, jac=True, method='L-BFGS-B', bounds=[(-12,4)]*(len(K)-1)+[(-10,-2.5)], options={'maxiter':100}).x
        w = np.exp(h)
        a = cho_solve(cho_factor((w@K).reshape(n,n)),v)
        columns.append(sum(wi*(g@(g[:n].T@a)) for wi,g in zip(w,design)))
    return np.column_stack(columns)


def _candidate(inputs, kind):
    data = _base_node(inputs)
    s = inputs['__input__']
    z, c = data['z'][:,:4], data['z'][:,4:]
    b = _mapped(s['panel_b'], z, 'v')
    pure = np.cos(2.4*z)
    if kind == 'base':
        return data
    if kind == 'unmixed':
        data['z'] = np.column_stack((z,c,pure))
    elif kind == 'affine_control':
        data['z'] = np.column_stack((z,c,b))
    elif kind == 'unmixed_interactions':
        cross = (z[:,:,None]*pure[:,None,:]).reshape(len(z),-1)
        data['z'] = np.column_stack((z,c,pure,cross))
    elif kind == 'rotated_interactions':
        cross = [b[:,i]*b[:,j] for i,j in combinations(range(4),2)]
        data['z'] = np.column_stack((z,c,b,*cross))
    else:
        learned = _profile_features(z,b,c,pure,data['y'],data['traj'],data['eval_times'])
        data['z'] = np.column_stack((z,c,learned))
    return data
