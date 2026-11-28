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




def _previous_node(inputs: dict) -> dict:
    data = _base_node(inputs)
    s = inputs['__input__']
    z, c = data['z'][:, :4], data['z'][:, 4:]
    b = _mapped(s['panel_b'], z, 'v')
    data['z'] = np.column_stack((z, c, b))
    return data
from itertools import combinations
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist


def _structured_features(data, observed):
    z = data['z']
    n = int(data['n_train'])
    times, traj = data['eval_times'], data['traj']

    def project(r):
        h = -np.expm1(-times[:, None] * r)
        return h, np.linalg.lstsq(h, traj.T, rcond=None)[0]

    def residual(r):
        h, m = project(r)
        return (h @ m - traj.T).ravel()

    rates = least_squares(residual, [.5, 2.], bounds=(.01, 10.)).x
    _, modes = project(np.sort(rates))
    targets = np.column_stack((data['y'], modes.T))
    covariances, cross, prior = [], [], []

    def add(k, log_weight):
        covariances.append(k[:n])
        cross.append(k)
        prior.append(log_weight)

    for q in (z[:, :4], z[:, 4:8], z[:, 8:12], observed):
        q = (q - q[:n].mean(0)) / np.maximum(q[:n].std(0), 1e-8)
        d = cdist(q, q[:n], 'sqeuclidean')
        linear = (q @ q[:n].T) / q.shape[1]
        add(linear, -2.5)
        for degree in (2, 3, 4):
            add((linear + 1)**degree / (2**degree - 1), -4.5)
        for length in (.55, .9, 1.5, 2.5, 4.):
            add(np.exp(-.5*d/length**2), -4.)
        for i, j in combinations(range(4), 2):
            a, b = q[:, i], q[:, j]
            for pair in (np.column_stack((a,b)), np.column_stack((a*b, a+b)), np.column_stack((a*b,a-b))):
                d = cdist(pair, pair[:n], 'sqeuclidean')
                for length in (.65, 1.15, 2.):
                    add(np.exp(-.5*d/length**2), -5.8)
        for v in (q.T if q.shape[1] == 4 else ()):
            d = v[:, None] - v[None, :n]
            for freq in (1., 2., 3., 4.):
                add(np.cos(freq*d), -5.5)
    eye = np.eye(n)
    train = np.vstack((np.asarray(covariances).reshape(-1,n*n), eye.ravel()))
    cross = np.asarray(cross).reshape(-1,len(z)*n)
    prior = np.r_[prior, -6.]
    results = []
    for target in targets.T:
        target = (target-target.mean()) / max(target.std(), 1e-8)
        def objective(logw):
            w = np.exp(logw)
            cf = cho_factor((w @ train).reshape(n,n))
            a = cho_solve(cf, target)
            g = cho_solve(cf, eye) - np.outer(a,a)
            delta = logw-prior
            return .5*target@a+np.log(np.diag(cf[0])).sum()+.10*(delta@delta), .5*w*(train@g.ravel())+.20*delta
        fit = minimize(objective, prior, jac=True, method='L-BFGS-B', bounds=[(-12,3)]*(len(prior)-1)+[(-10,-2.4)], options={'maxiter':80})
        w = np.exp(fit.x)
        a = cho_solve(cho_factor((w@train).reshape(n,n)),target)
        results.append((w[:-1]@cross).reshape(len(z),n)@a)
    return np.column_stack(results)


def _transfer(inputs, scale):
    data = _previous_node(inputs)
    study = inputs['__input__']
    observed = np.vstack((study['train']['x'],study['test']['x']))
    specialized = _structured_features(data, observed)
    data['z'] = np.column_stack((data['z'], np.tile(specialized, (1, scale))))
    return data
