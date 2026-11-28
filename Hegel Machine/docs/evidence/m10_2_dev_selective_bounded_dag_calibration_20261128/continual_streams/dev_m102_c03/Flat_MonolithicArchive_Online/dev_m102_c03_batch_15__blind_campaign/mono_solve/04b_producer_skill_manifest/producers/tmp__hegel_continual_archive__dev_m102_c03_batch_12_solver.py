"""Invert the calibration and predict from the measured dual branch and its interactions."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel, x):
    u, v = panel['u'], panel['v']
    a = _aff(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(5):
        err = np.linalg.norm(v - a @ _ls(a[keep], v[keep]), axis=1)
        keep = err < max(.005, 3*np.median(err))
    affine = _ls(a[keep], v[keep])
    ident = np.eye(u.shape[1])

    def design(z, rates):
        sine_rate, tanh_rate = rates
        eye = np.broadcast_to(ident, (len(z),) + ident.shape)
        return np.concatenate((eye, z[:, :, None]*eye, np.tanh(tanh_rate*z)[:, :, None],
                               np.sin(sine_rate*np.roll(z, 2, axis=1))[:, :, None]), axis=2)

    for _ in range(3):
        def fit(rate):
            d = design(u[keep], rate).reshape(-1, 10)
            target = v[keep].ravel()
            coef = _ls(d, target)
            return d@coef-target, coef
        rate = least_squares(lambda r: fit(r)[0], [2.4, 1.8],
                             bounds=([1.4, .4], [3.5, 4.0]), gtol=1e-11).x
        coef = fit(rate)[1]
        err = np.linalg.norm(design(u, rate) @ coef-v, axis=1)
        new = err < max(.001, 4*np.median(err))
        if new.sum() < 20 or np.array_equal(new, keep):
            break
        keep = new
    init = _ls(affine[1:].T, (x-affine[0]).T).T
    return np.array([least_squares(lambda z: design(z[None], rate)[0]@coef-row,
                                   z0, gtol=1e-11).x for row, z0 in zip(x, init)])


def _features(study, x):
    z = _calibration(study['panel_a'], x)
    u, v1, v2 = (study['panel_c'][k] for k in ('u','v1','v2'))
    def cyc(w):
        return _aff(w*np.roll(w,-1,axis=1))
    def cos(w, r):
        return _aff(np.cos(r*w))
    def fit(r):
        a = cos(u,r)
        b = _ls(a,v2)
        return np.mean((a@b-v2)**2), b
    r = minimize_scalar(lambda r: fit(r)[0], bounds=(1.8,2.9), method='bounded').x
    c = cyc(z)@_ls(cyc(u),v1)
    d = cos(z,r)@fit(r)[1]
    return np.column_stack((_aff(z), c, d, (c[:,:,None]*d[:,None,:]).reshape(len(z),16)))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    a = _features(study, np.vstack((train['x'],study['test']['x'])))
    c, d = a[:, 5:9], a[:, 9:13]
    cross = np.sum(c*d, axis=1)
    scalar = _aff(np.column_stack((c, d, cross)))
    spatial = np.column_stack((scalar, c[:, 0]*d[:, 1]))
    y_hat = scalar[n:] @ _ls(scalar[:n], train['y'])
    times = np.asarray(study['eval_times'])
    traj = train['traj'].T

    def modes(r):
        return -np.expm1(-times[:, None]*r)

    def time_error(r):
        f = modes(r)
        return (f @ _ls(f, traj) - traj).ravel()

    rates = least_squares(time_error, [.4, 1.8], bounds=([.05, .9], [.85, 4])).x
    f = modes(rates)
    state_pred = spatial[n:] @ _ls(spatial[:n], _ls(f, traj).T) @ f.T
    last = state_pred[:, -1]
    return dict(y_hat=y_hat, state_pred=state_pred,
                diagnostics_vec=np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]))
