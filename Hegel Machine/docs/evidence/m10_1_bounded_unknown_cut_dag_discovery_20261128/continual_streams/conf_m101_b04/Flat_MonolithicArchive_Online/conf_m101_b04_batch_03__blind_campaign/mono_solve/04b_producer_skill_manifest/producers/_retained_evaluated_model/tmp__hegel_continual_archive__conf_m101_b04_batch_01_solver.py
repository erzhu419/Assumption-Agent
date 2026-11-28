"""Panel-aligned nonlinear response and low-rank relaxation forecasts."""
from itertools import combinations, combinations_with_replacement
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, j]*x[:, k] for j,k in combinations_with_replacement(range(4), 2))))


def _coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    keep = np.arange(len(u))
    for iteration in range(3):
        beta = np.array([_fit(_affine(u[keep, j:j+1]), v[keep, j]) for j in range(4)])
        residual = np.linalg.norm(v - (beta[:, 0] + u * beta[:, 1]), axis=1)
        keep = (np.argsort(residual)[:int(.8*len(u))] if iteration == 0 else
                np.flatnonzero(residual < max(.001, 8*np.median(residual))))
    z = beta[:, 0] + x * beta[:, 1]
    b = study['panel_b']
    return z, _affine(z) @ _fit(_affine(b['u']), b['v'])


def _bank(x, a, b):
    columns = []
    for z in (x, a, b):
        dirs = [z[:, j] for j in range(4)]
        dirs.extend(z[:, j] + c*z[:, k] for j,k in combinations(range(4),2) for c in (-2,-1,-.5,.5,1,2))
        for j, v in enumerate(dirs):
            for k in (.6,1,1.6,2.2,3.2):
                w = k*v
                for f in (np.sin(w), np.cos(w), np.tanh(w), 1/(1+w*w), v/(1+w*w)):
                    columns.append(f)
                    if j < 4:
                        for m in range(4):
                            columns.extend((f*z[:,m], f*a[:,m]))
        for degree in (3,4):
            columns.extend(np.prod(z[:, ix], axis=1) for ix in combinations_with_replacement(range(4), degree))
    return np.column_stack(columns)


def _selected(q, f, target, count=10):
    l = np.linalg.qr(q, mode='reduced')[0]
    g = f - l@(l.T@f)
    scale = np.linalg.norm(g, axis=0)
    g /= np.where(scale > 1e-6, scale, np.inf)
    r = target - l@(l.T@target)
    error = r
    chosen = []
    for _ in range(count):
        score = np.sum((g.T @ error)**2, axis=1)
        score[chosen] = -1
        chosen.append(np.argmax(score))
        active = g[:,chosen]
        error = r - active@_fit(active, r)
    return chosen


def _smooth(q, qt, k, kt, y, penalty):
    factor = cho_factor(k + penalty*np.eye(len(k)))
    aq, ay = cho_solve(factor, q), cho_solve(factor, y)
    beta = np.linalg.solve(q.T @ aq, q.T @ ay)
    return qt@beta + kt@(ay-aq@beta)


def _trajectory(q, qt, k, kt, traj, t):
    def basis(rates):
        return -np.expm1(-t[:, None]*rates)

    def residual(rates):
        b = basis(rates)
        return (b@_fit(b, traj.T)-traj.T).ravel()

    rates = least_squares(residual, [1.5,.25], bounds=([.7,.02],[4.,.65])).x
    modes = np.linalg.qr(basis(rates), mode='reduced')[0]
    amps = _smooth(q, qt, k, kt, traj@modes, .003)
    return amps@modes.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt, y, traj = train['x'], study['test']['x'], train['y'], train['traj']
    q, qt = _quadratic(x), _quadratic(xt)
    a, b = _coordinates(study, np.vstack((x, xt)))
    f = _bank(np.vstack((x, xt)), a, b)
    n = len(x)
    target = np.column_stack((y, traj[:, -3:].mean(1)))

    def predict(chosen):
        design = np.column_stack((q, f[:n, chosen]))
        design_test = np.column_stack((qt, f[n:, chosen]))
        return design_test @ _fit(design, y)

    full = predict(_selected(q, f[:n], target, count=8))
    rng = np.random.default_rng(62)
    predictions = []
    for _ in range(63):
        rows = rng.choice(n, int(.83*n), replace=False)
        chosen = _selected(q[rows], f[rows], target[rows], count=6)
        predictions.append(predict(chosen))
    # Average across feature sets to reduce sensitivity to the training sample.
    yh = .5 * (full + np.mean(predictions, axis=0))
    k, kt = (np.exp(-.23*cdist(w, x, 'sqeuclidean')) for w in (x, xt))
    th = _trajectory(q, qt, k, kt, traj, study['eval_times'])
    return {'y_hat': yh, 'state_pred': th,
            'diagnostics_vec': np.array([yh.mean(), yh.std(), th[:,-1].mean(), th[:,-1].std()])}
