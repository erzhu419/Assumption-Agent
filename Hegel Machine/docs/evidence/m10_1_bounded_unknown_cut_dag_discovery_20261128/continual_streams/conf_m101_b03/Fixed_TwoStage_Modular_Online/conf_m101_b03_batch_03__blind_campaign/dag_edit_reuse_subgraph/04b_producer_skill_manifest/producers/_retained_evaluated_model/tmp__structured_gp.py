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
