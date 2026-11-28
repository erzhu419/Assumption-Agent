"""Panel-calibrated scalar and trajectory prediction."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf, hyp2f1


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    du = u[j] - u[i]
    g = np.array([np.median(((v[j] - v[i])[:, k] / np.where(du[:, k] == 0, np.nan, du[:, k]))[np.abs(du[:, k]) > 0.05]) for k in range(u.shape[1])])
    b = np.median(v - u*g, axis=0)
    err = np.linalg.norm(v - u*g - b, axis=1)
    keep = err < 6*np.median(err)
    u, v = u[keep], v[keep]
    c = u - u.mean(axis=0)
    gain = (c*v).sum(axis=0) / (c*c).sum(axis=0)
    return (x - v.mean(axis=0)) / gain + u.mean(axis=0)


def _panel_b_basis(u, scale):
    return np.column_stack((_affine(u), erf(scale*u)))


def _panel_c_basis(u, scale, power):
    return np.column_stack((_affine(u), u*hyp2f1(0.5, power, 1.5, -(scale*u)**2)))


def _quadratic(u):
    i, j = np.triu_indices(u.shape[1])
    return np.column_stack((_affine(u), u[:, i]*u[:, j]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = _calibrate(study['panel_a'], np.concatenate((train['x'], study['test']['x'])))
    y, traj = train['y'], train['traj']
    times = np.asarray(study['eval_times'])
    b, c = study['panel_b'], study['panel_c']

    def c_residual(p):
        d = _panel_c_basis(c['u'], *p)
        return (d @ _fit(d, c['v1']) - c['v1']).ravel()

    c_shape = least_squares(c_residual, [7., .065], bounds=([2., .016], [14., .4])).x
    c_weights = _fit(_panel_c_basis(c['u'], *c_shape), c['v1'])

    def b_error(scale):
        d = _panel_b_basis(b['u'], scale)
        return np.mean((d @ _fit(d, b['v']) - b['v'])**2)

    scale = minimize_scalar(b_error, bounds=(.7, 2.1), method='bounded').x
    b_weights = _fit(_panel_b_basis(b['u'], scale), b['v'])

    def design(p, xx):
        z = _panel_b_basis(xx, p[0]) @ p[1:-2].reshape(b_weights.shape)
        q = _panel_c_basis(z, *c_shape) @ c_weights
        modes = -np.expm1(-np.exp(p[-2:])[:, None]*times)
        return _quadratic(q), _affine(q), modes

    def residual(p):
        f, g, h = design(p, x[:n])
        ey = f @ _fit(f, y) - y
        et = g @ _fit(g, traj @ np.linalg.pinv(h)) @ h - traj
        ep = _panel_b_basis(b['u'], p[0]) @ p[1:-2].reshape(b_weights.shape) - b['v']
        # Anchor the operator with the reference panel while fitting both outputs.
        return np.r_[ey/.014, et.ravel()/.018, ep.ravel()/.00275]

    p0 = np.r_[scale, b_weights.ravel(), np.log([.415, 1.81])]
    lo = np.r_[.7, np.full(b_weights.size, -np.inf), -1.7, .15]
    hi = np.r_[2.3, np.full(b_weights.size, np.inf), -.25, 1.3]
    p = least_squares(residual, p0, bounds=(lo, hi), max_nfev=24).x
    f, g, h = design(p, x)
    py = f[n:] @ _fit(f[:n], y)
    pt = g[n:] @ _fit(g[:n], traj @ np.linalg.pinv(h)) @ h
    return {'y_hat': py, 'state_pred': pt,
            'diagnostics_vec': np.array([py.mean(), py.std(), pt[:, -1].mean(), pt[:, -1].std()])}
