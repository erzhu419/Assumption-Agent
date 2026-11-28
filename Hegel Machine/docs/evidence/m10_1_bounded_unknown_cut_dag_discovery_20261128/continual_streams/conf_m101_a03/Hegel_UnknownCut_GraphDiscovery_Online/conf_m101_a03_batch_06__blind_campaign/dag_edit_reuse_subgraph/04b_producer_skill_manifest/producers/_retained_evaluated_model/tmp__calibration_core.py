import numpy as np


def _features(u):
    n, d = u.shape
    channel = np.broadcast_to(np.eye(d), (n, d, d))
    neighbor = np.roll(u, -1, axis=1)
    return np.concatenate((channel, channel * u[..., None],
                           neighbor[..., None], (u * neighbor)[..., None],
                           (u ** 3)[..., None]), axis=2).reshape(n * d, 2 * d + 3)


def _fit(u, v):
    design = _features(u)
    y = v.ravel()
    d = u.shape[1]
    weights = np.ones(len(u))
    for scale in (0.15, 0.05, 0.015, 0.004, 0.0015, 0.0015):
        w = np.repeat(weights, d)
        p = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
        residual = np.linalg.norm((design @ p).reshape(v.shape) - v, axis=1)
        weights = 1 / np.hypot(1, residual / scale)
    keep = np.repeat(np.argsort(np.argsort(residual)) < len(u) - 3, d)
    return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]


def _invert(v, p):
    d = v.shape[1]
    offset, gain = p[:d], p[d:2*d]
    coupling, interaction, cubic = p[-3:]
    j = np.arange(d)
    nxt = (j + 1) % d
    linear = np.diag(gain)
    linear[j, nxt] = coupling
    x = np.linalg.solve(linear, (v - offset).T).T

    def error(x):
        neighbor = x[:, nxt]
        return offset + gain * x + (coupling + interaction * x) * neighbor + cubic * x ** 3 - v

    for _ in range(25):
        r = error(x)
        if np.max(np.abs(r), initial=0.0) < 1e-11:
            break
        jac = np.zeros((len(x), d, d))
        jac[:, j, j] = gain + interaction * x[:, nxt] + 3 * cubic * x ** 2
        jac[:, j, nxt] = coupling + interaction * x
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        trial = x - step
        baseline = np.linalg.norm(r, axis=1)
        for _ in range(12):
            worse = np.linalg.norm(error(trial), axis=1) > baseline
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = x[worse] - step[worse]
        x = trial
    return x


from itertools import combinations_with_replacement
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _polynomial(x, degree):
    return np.column_stack([np.prod(x[:, q], axis=1)
                            for k in range(degree + 1)
                            for q in combinations_with_replacement(range(x.shape[1]), k)])


def _index(d, y, n, mode):
    a = d[:n]
    if mode == 0:
        # A cubic index with a bounded nonlinear correction of that same index.
        initial = np.linalg.lstsq(a, y, rcond=None)[0]
        penalty = np.r_[np.zeros(15), np.full(20, np.sqrt(0.002))]
        def predict(p, x):
            z = x @ p[:-3]
            return z + p[-2] * np.tanh(p[-1] * z + p[-3])
        fits = [least_squares(lambda p: np.r_[predict(p, a)-y, penalty*p[:-3]],
                             np.r_[initial, 0., .2, f], max_nfev=100,
                             bounds=(np.r_[np.full(37, -np.inf), .2], np.r_[np.full(37, np.inf), 4.]))
                for f in (1., 2.5)]
        return predict(min(fits, key=lambda p: p.cost).x, d)

    # Profile the linear index out of the bounded additive correction.
    # Penalize cubic terms more than quadratic terms to stabilize extrapolation.
    degree = 2 if mode in (1, 2) else 3
    m = 15 if degree == 2 else 35
    a = a[:, :m]
    interaction = d[:, :15 if mode in (2, 4) else 5]
    b = interaction[:n]
    reg = np.r_[np.zeros(5), np.full(10, .003), np.full(m-15, .035)]
    ext = np.vstack((a, np.diag(np.sqrt(reg))))
    pinv = np.linalg.pinv(ext)
    target = np.r_[y, np.zeros(m)]
    py = target - ext @ (pinv @ target)
    project = np.eye(n+m) - ext @ pinv
    proj = project[:, :n]
    shrink = np.r_[.002, np.full(4, .012), np.full(b.shape[1]-5, .040)]
    def fun(p):
        h = np.tanh(b @ p[:-1])
        return np.r_[py - p[-1]*(proj@h), shrink*p[:-1]]
    def jac(p):
        h = np.tanh(b @ p[:-1]); g = np.empty((n+m+len(shrink), len(p)))
        g[:n+m, :-1] = -p[-1]*(proj @ ((1-h*h)[:,None]*b))
        g[:n+m, -1] = -(proj@h)
        g[n+m:] = 0.; g[n+m:, :-1] = np.diag(shrink)
        return g
    base = np.linalg.lstsq(b, y, rcond=None)[0]
    rng = np.random.default_rng(103)
    starts = [base, -base, *rng.normal(0, .65, size=(5,len(base)))]
    fits = [least_squares(fun, np.r_[q, .22], jac=jac, max_nfev=75,
                          bounds=(np.r_[np.full(len(q), -np.inf), -.65], np.r_[np.full(len(q), np.inf), .65]))
            for q in starts]
    p = min(fits, key=lambda w:w.cost).x
    correction = p[-1]*np.tanh(interaction@p[:-1])
    coef = pinv @ (target - np.r_[correction[:n], np.zeros(m)])
    return d[:, :m] @ coef + correction


def _coordinates(inputs, mode, chart):
    study = inputs['__input__']; panel = study['panel_a']
    train = study['train']; n = len(train['x'])
    x = _invert(np.vstack((train['x'], study['test']['x'])), _fit(panel['u'], panel['v']))
    b = study['panel_b']; operator = np.linalg.lstsq(_affine(b['u']), b['v'], rcond=None)[0]
    z = _affine(x) @ operator
    d = _polynomial(x, 3)
    trajectory = train['traj']
    centered = trajectory - trajectory.mean(axis=0)
    modes = np.linalg.svd(centered, full_matrices=False)[2][:2]
    scores = centered @ modes.T
    # Both time modes share the same quadratic coordinate. Estimate that
    # coordinate jointly rather than fitting independent interactions.
    q = d[:, 5:15]; a = d[:, :5]
    independent = q[:n] - a[:n] @ np.linalg.lstsq(a[:n], q[:n], rcond=None)[0]
    weights = np.linalg.lstsq(independent, scores, rcond=None)[0]
    left, _, _ = np.linalg.svd(weights, full_matrices=False)
    f = np.column_stack((a, q @ left[:, 0]))
    scores = f @ np.linalg.lstsq(f[:n], scores, rcond=None)[0]
    signal = _index(d, train['y'], n, mode)
    extra = z[:, 0] if chart == 0 else z[:, 1] if chart == 1 else z[:, 3]
    target = np.column_stack((signal, scores, extra))
    target = (target - target[:n].mean(axis=0)) / np.maximum(target[:n].std(axis=0), 1e-8)
    calibrated = np.linalg.solve(operator[1:].T, (target - operator[0]).T).T
    return {'x_cal_tr': calibrated[:n], 'x_cal_te': calibrated[n:]}
