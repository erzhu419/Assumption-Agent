"""Calibrated dual-operator predictor with a two-rate state response."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(*cols):
    return np.column_stack((np.ones(len(cols[0])), *cols))


def _calibrate(panel, observed):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.max(np.abs(v-u), axis=1) < 3
    u, v = u[keep], v[keep]
    r = np.roll(u, -1, axis=1)
    c = np.column_stack([_ls(_aff(u[:, j], r[:, j], u[:, j]*r[:, j], u[:, j]**3), v[:, j])
                         for j in range(u.shape[1])])
    offset, gain, coupling, product, cubic = c
    z = (observed-offset)/gain
    j = np.arange(z.shape[1]); neighbor = (j+1) % len(j)
    for _ in range(45):
        r = z[:, neighbor]
        err = offset+gain*z+coupling*r+product*z*r+cubic*z**3-observed
        if np.max(np.abs(err), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = gain+product*r+3*cubic*z*z
        jac[:, j, neighbor] = coupling+product*z
        z -= np.clip(np.linalg.solve(jac, err[..., None])[..., 0], -.4, .4)
    return z


def _operators(panel, z):
    u = np.asarray(panel['u'], dtype=float)
    i, j = np.triu_indices(u.shape[1], 1)
    def design(x, k, mode):
        h = x/(1+k*np.abs(x)) if mode == 1 else x/np.sqrt(1+k*x*x)
        return _aff(x, h, x[:, i]*x[:, j])
    prediction = []
    for mode in (1, 2):
        target = np.asarray(panel[f'v{mode}'], dtype=float)
        def fit(k):
            a = design(u, k, mode)
            w = _ls(a, target)
            return np.mean((a@w-target)**2), w
        k = minimize_scalar(lambda k: fit(k)[0], bounds=(.1, 2.), method='bounded').x
        prediction.append(design(z, k, mode)@fit(k)[1])
    return prediction


def _basis(times, rates):
    return -np.expm1(-np.outer(times, rates))


def _rates(times, traj):
    def residual(rates):
        b = _basis(times, rates)
        return (b@_ls(b, traj.T)-traj.T).ravel()
    return least_squares(residual, [1.8, .43], bounds=([.8, .03], [8., .79])).x


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    p, q = _operators(study['panel_c'], _calibrate(study['panel_a'], x))
    pq = (p[:, :, None]*q[:, None, :]).reshape(len(x), -1)
    base = _aff(p, q, np.sum(p*q, axis=1))
    traj = np.asarray(train['traj'], dtype=float)
    times = np.asarray(study['eval_times'], dtype=float)
    b = _basis(times, _rates(times, traj))
    target = np.column_stack((train['y'], _ls(b, traj.T).T))
    scale = np.maximum(target.std(axis=0), 1e-12)
    def fit_candidate(j):
        a = np.column_stack((base, pq[:, j]))
        w = _ls(a[:n], target)
        return np.mean(((a[:n]@w-target)/scale)**2), a[n:]@w
    _, prediction = min((fit_candidate(j) for j in range(pq.shape[1])), key=lambda x: x[0])
    y_hat = prediction[:, 0]
    state = prediction[:, 1:]@b.T
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), state[:, -1].mean(), state[:, -1].std()])}
