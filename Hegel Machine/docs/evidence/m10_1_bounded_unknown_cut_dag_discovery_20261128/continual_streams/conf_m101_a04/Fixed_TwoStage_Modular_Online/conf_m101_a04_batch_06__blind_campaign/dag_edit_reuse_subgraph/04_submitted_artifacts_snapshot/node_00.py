import numpy as np
from scipy.optimize import minimize_scalar


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate_a(panel, measured):
    u, v = panel['u'], panel['v']
    nxt = np.roll(u, -1, axis=1)
    coef = np.empty((4, u.shape[1]))
    for j in range(u.shape[1]):
        x, t = u[:, j], nxt[:, j]
        b = np.column_stack((np.ones(len(u)), x, t, x*t + 0.5*x**3))
        coef[:, j] = np.linalg.lstsq(b, v[:, j], rcond=None)[0]
    for _ in range(16):
        residual = v - (coef[0] + coef[1]*u + coef[2]*nxt + coef[3]*(u*nxt + 0.5*u**3))
        norms = np.linalg.norm(residual, axis=1)
        scale = max(np.median(norms), 2e-4)
        w = 1/np.sqrt(1+(norms/scale)**2)
        for j in range(u.shape[1]):
            x, t = u[:, j], nxt[:, j]
            b = np.column_stack((np.ones(len(u)), x, t, x*t + 0.5*x**3))
            coef[:, j] = np.linalg.lstsq(b*w[:, None], v[:, j]*w, rcond=None)[0]
    x = np.linalg.solve((np.diag(coef[1]) + np.roll(np.diag(coef[2]), 1, axis=1)), (measured-coef[0]).T).T
    idx = np.arange(x.shape[1])
    for _ in range(24):
        t = np.roll(x, -1, axis=1)
        r = coef[0] + coef[1]*x + coef[2]*t + coef[3]*(x*t + 0.5*x**3) - measured
        if np.max(np.abs(r)) < 1e-11:
            break
        jac = np.zeros((len(x), x.shape[1], x.shape[1]))
        jac[:, idx, idx] = coef[1] + coef[3]*(t + 1.5*x*x)
        jac[:, idx, (idx+1)%len(idx)] = coef[2] + coef[3]*x
        x -= np.linalg.solve(jac, r[..., None])[..., 0]
    return x


def _features(study):
    raw = np.concatenate((study['train']['x'], study['test']['x']))
    x = _calibrate_a(study['panel_a'], raw)
    b = study['panel_b']
    bv = _design(x) @ np.linalg.lstsq(_design(b['u']), b['v'], rcond=None)[0]
    c = study['panel_c']
    u = c['u']
    v1 = _design(x*np.roll(x, -1, axis=1)) @ np.linalg.lstsq(
        _design(u*np.roll(u, -1, axis=1)), c['v1'], rcond=None)[0]

    def cosine_fit(freq):
        basis = _design(np.cos(freq*u))
        coef = np.linalg.lstsq(basis, c['v2'], rcond=None)[0]
        return np.sum((basis@coef-c['v2'])**2), coef

    freq = minimize_scalar(lambda f: cosine_fit(f)[0], bounds=(1.5, 3.5), method='bounded').x
    v2 = _design(np.cos(freq*x)) @ cosine_fit(freq)[1]
    return x, bv, v1, v2


def _pair_products(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def _project(basis, target, penalty):
    """Fit a standardized ridge projection using only the training rows."""
    n = len(target)
    center = basis[:n].mean(axis=0)
    scale = np.maximum(basis[:n].std(axis=0), 1e-12)
    a = (basis - center) / scale
    tr = a[:n]
    coef = np.linalg.solve(tr.T @ tr + np.diag(penalty),
                           tr.T @ (target - target.mean(axis=0)))
    return a @ coef


def _readout_features(b, c, d, y, traj):
    shared = np.column_stack((b, c, d, _pair_products(b, c)))
    scalar_basis = np.column_stack((shared, c*d))
    sy = _project(scalar_basis, y, np.full(scalar_basis.shape[1], .002))

    # The trajectory mixes the diagonal contraction with an off-diagonal
    # v1-v2 interaction. Keep the remaining pairs available with shrinkage.
    pairs = _pair_products(c, d)
    ix = np.arange(c.shape[1])
    diag = ix*c.shape[1] + ix
    off = np.setdiff1d(np.arange(pairs.shape[1]), np.r_[diag, 1])
    primary = np.column_stack((shared, (c*d).sum(axis=1), pairs[:, 1]))
    other = np.column_stack((c[:, :-1]*d[:, :-1] - c[:, -1:]*d[:, -1:],
                             pairs[:, off]))
    basis = np.column_stack((primary, other))
    penalty = np.r_[np.full(primary.shape[1], .002), np.full(other.shape[1], .3)]
    _, _, modes = np.linalg.svd(traj - traj.mean(axis=0), full_matrices=False)
    st = _project(basis, traj @ modes[:2].T, penalty)
    h = np.column_stack((sy, st))
    tr = h[:len(y)]
    return .25 * (h - tr.mean(axis=0)) / np.maximum(tr.std(axis=0), 1e-12)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    _, b, c, d = _features(study)
    h = _readout_features(b, c, d, train['y'], train['traj'])
    return {'h': h, 'n_train': len(train['x']), 'y': train['y'], 'traj': train['traj']}
