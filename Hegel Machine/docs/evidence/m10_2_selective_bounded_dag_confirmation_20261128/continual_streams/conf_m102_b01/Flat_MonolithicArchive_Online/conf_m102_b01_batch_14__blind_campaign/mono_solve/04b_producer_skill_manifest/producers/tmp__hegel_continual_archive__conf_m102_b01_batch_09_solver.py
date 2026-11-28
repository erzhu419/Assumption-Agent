import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, observations):
    # Calibrate each channel independently so a corrupted panel entry does not
    # change the fitted calibration of the other channels.
    u, v = panel['u'], panel['v']
    basis = [_aff(np.column_stack((u[:, j], np.tanh(1.8*u[:, j]),
                                    np.sin(2.4*u[:, (j+2) % 4])))) for j in range(4)]
    coeff = []
    for j, d in enumerate(basis):
        response = v[:, j]
        rows = np.arange(len(d))
        for _ in range(9):
            w = _ls(d[rows], response[rows])
            residual = np.abs(d @ w - response)
            rows = np.argsort(residual)[:int(len(d)*0.84)]
        inlier = residual < max(0.0007, 6*np.median(residual))
        coeff.append(_ls(d[inlier], response[inlier]))
    bias, gain, sat, cross = np.asarray(coeff).T

    def forward(q):
        return bias + gain*q + sat*np.tanh(1.8*q) + cross*np.sin(2.4*np.roll(q, 2))

    def inverse(r):
        guess = (r-bias)/gain
        opt = root(lambda q: forward(q)-r, guess, tol=1e-10)
        if np.max(np.abs(forward(opt.x)-r)) > 1e-8:
            return least_squares(lambda q: forward(q)-r, guess).x
        return opt.x

    return np.asarray([inverse(r) for r in observations]).reshape(-1, 4)


def _coordinates(study, x):
    z = _calibrate(study['panel_a'], x)
    b = study['panel_b']
    w = _aff(z) @ _ls(_aff(b['u']), b['v'])
    c = study['panel_c']

    def features(r, scale):
        return _aff(np.column_stack((r, np.arcsinh(scale*r))))

    def loo(scale):
        d = features(c['u'], scale)
        inv = np.linalg.pinv(d)
        residual = d @ (inv @ c['v1']) - c['v1']
        lever = np.sum(d*inv.T, axis=1)
        return np.mean((residual/(1-lever[:, None]))**2)

    scale = minimize_scalar(loo, bounds=(1., 6.), method='bounded').x
    return features(w, scale) @ _ls(features(c['u'], scale), c['v1'])


def _quadratic(q):
    return _aff(np.column_stack([q] + [q[:, i]*q[:, j] for i in range(4) for j in range(i, 4)]))


def _trajectory(q, traj, t, qt):
    def time_features(rate):
        return -np.expm1(-t[:, None]*rate)

    def objective(rate):
        d = time_features(rate)
        return (d @ _ls(d, traj.T)-traj.T).ravel()

    r = least_squares(objective, [1.8, .4], bounds=([1.0, .02], [4., .99])).x
    d = time_features(r)
    amplitudes = _ls(d, traj.T).T
    return (_aff(qt) @ _ls(_aff(q), amplitudes)) @ d.T


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    q = _coordinates(study, np.concatenate((train['x'], test['x'])))
    q, qt = q[:n], q[n:]
    y = _quadratic(qt) @ _ls(_quadratic(q), train['y'])
    state = _trajectory(q, train['traj'], np.asarray(study['eval_times']), qt)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
