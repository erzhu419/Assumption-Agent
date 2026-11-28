"""Calibrate the input, compose the active panel operators, and predict two readouts."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    corrected = np.empty_like(x, dtype=float)
    for j in range(u.shape[1]):
        d = np.column_stack((np.ones(len(u)), u[:, j]))
        keep = np.ones(len(u), bool)
        for _ in range(8):
            offset, gain = np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]
            r = v[:, j] - offset - gain*u[:, j]
            next_keep = abs(r) <= max(.0006, 4*np.median(abs(r)))
            if next_keep.sum() < 3 or np.array_equal(next_keep, keep):
                break
            keep = next_keep
        corrected[:, j] = (x[:, j]-offset)/gain
    return corrected


def _operator(panel, key, x, kind):
    u, v = (np.asarray(panel[k], float) for k in ('u', key))
    def features(q, p):
        k, a, b = p[:3]
        if kind == 'b':
            z = erf(np.exp(k)*q + a*q**3) + b*q**3 + p[3]*q
        else:
            z = q*abs(q)/(1+np.exp(k)*abs(q)) + a*q**3 + b*q
        return np.column_stack((np.ones(len(q)), z))

    def fit(p):
        d = features(u, p)
        coef = np.linalg.lstsq(d, v, rcond=None)[0]
        return (d@coef-v).ravel(), coef

    initial = [.63, 4., -.62, -2.13] if kind == 'b' else [-.42, -.02, -1.08]
    pars = least_squares(lambda p: fit(p)[0], initial, max_nfev=200).x
    return features(x, pars)@fit(pars)[1]


def _coordinates(study, x):
    x = _calibrate(study['panel_a'], x)
    b = _operator(study['panel_b'], 'v', x, 'b')
    c = _operator(study['panel_c'], 'v1', b, 'c')
    return x, b, c


def _design(x, c):
    return np.column_stack((np.ones(len(x)), x, c,
                            (x[:, :, None]*c[:, None, :]).reshape(len(x), 16)))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x, _, c = _coordinates(study, np.concatenate((train['x'], study['test']['x'])))
    d = _design(x/x[:n].std(0), c/c[:n].std(0))
    t = np.asarray(study['eval_times'])
    traj = np.asarray(train['traj'])

    def time_basis(rates):
        return np.column_stack((-np.expm1(-rates[0]*t), t/(1+rates[1]*t)))

    def time_residual(rates):
        basis = time_basis(rates)
        return (basis @ np.linalg.lstsq(basis, traj.T, rcond=None)[0] - traj.T).ravel()

    rates = least_squares(time_residual, [1.8, .4], bounds=([.2, .05], [5., 2.])).x
    time = time_basis(rates)
    target = np.column_stack((train['y'], np.linalg.lstsq(time, traj.T, rcond=None)[0].T))
    a = d[:n]
    w = np.linalg.solve(a.T@a+.003*np.eye(a.shape[1]), a.T@target)
    pred = d[n:]@w
    y, state = pred[:, 0], pred[:, 1:]@time.T
    return dict(y_hat=y, state_pred=state, diagnostics_vec=np.array(
        [y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()]))
