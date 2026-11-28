"""Calibrate the observed coordinates and fit the two response readouts."""
import numpy as np
from scipy.optimize import least_squares


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, observed):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    affine = _design(u)
    err = np.linalg.norm(v - affine @ _ls(affine, v), axis=1)
    good = err <= 4 * max(np.median(err), 1e-8)
    nxt = np.roll(np.arange(u.shape[1]), -1)
    designs = [_design(np.column_stack((u[:, j], u[:, k], u[:, j]*u[:, k], u[:, j]**3)))
               for j, k in enumerate(nxt)]
    for _ in range(4):
        coef = np.array([_ls(d[good], v[good, j]) for j, d in enumerate(designs)]).T
        residual = np.column_stack([d @ coef[:, j] - v[:, j] for j, d in enumerate(designs)])
        err = np.linalg.norm(residual, axis=1)
        updated = err <= max(1e-7, 6 * np.median(err))
        if np.array_equal(updated, good) or updated.sum() < 6:
            break
        good = updated
    offset, scale, coupling, pair, cubic = coef
    z = (observed - offset) / scale
    rows = np.arange(len(scale))

    def residual(z):
        return offset + scale*z + (coupling + pair*z)*z[:, nxt] + cubic*z**3 - observed

    r = residual(z)
    for _ in range(35):
        norm = np.sum(r*r, axis=1)
        if np.max(norm, initial=0.) < 1e-24:
            break
        jac = np.zeros((len(z), len(scale), len(scale)))
        jac[:, rows, rows] = scale + pair*z[:, nxt] + 3*cubic*z*z
        jac[:, rows, nxt] = coupling + pair*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        factor = np.ones((len(z), 1))
        candidate = z - step
        for _ in range(15):
            rc = residual(candidate)
            worse = np.sum(rc*rc, axis=1) > norm
            if not np.any(worse):
                break
            factor[worse] *= .5
            candidate[worse] = z[worse] - factor[worse]*step[worse]
        z = candidate
        r = residual(z)
    return z


def _branch(panel, z, k):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'+str(k)])

    def basis(x, p):
        w = p[0]*x + p[1:]
        nonlinear = w/(1+np.abs(w)) if k == 1 else np.cos(w)
        return _design(np.column_stack((x, nonlinear)))

    def residual(p):
        d = basis(u, p)
        return (d @ _ls(d, v) - v).ravel()

    start = np.r_[.84 if k == 1 else 1.27, np.zeros(u.shape[1])]
    p = least_squares(residual, start, max_nfev=150).x
    return basis(z, p) @ _ls(basis(u, p), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['x'])
    z = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    c, d = (_branch(study['panel_c'], z, k) for k in (1, 2))
    y_design = _design(np.column_stack((d, c, np.sum(c*d, axis=1))))
    cross = (d[:, :, None] * c[:, None, :]).reshape(len(z), -1)
    traj_design = _design(np.column_stack((d, c, cross)))
    return {'y_hat': y_design[n:] @ _ls(y_design[:n], train['y']),
            'unprojected': traj_design[n:] @ _ls(traj_design[:n], train['traj'])}
