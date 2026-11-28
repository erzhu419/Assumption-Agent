"""Calibrated dual-panel scalar prediction and independent trajectory fit."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    n = len(u)
    a = u.ravel()
    b = np.roll(u, -1, axis=1).ravel()
    channel = np.tile(np.eye(4), (n, 1))
    design = np.column_stack((channel, channel * a[:, None], b, a*b, a**3))
    discrepancy = np.linalg.norm(v-u, axis=1)
    good = discrepancy < 4 * max(np.median(discrepancy), 1e-8)
    if good.sum() < 8:
        good[:] = True
    for _ in range(5):
        keep = np.repeat(good, 4)
        coef = _fit(design[keep], v.ravel()[keep])
        residual = np.linalg.norm((design @ coef).reshape(n, 4) - v, axis=1)
        refined = residual <= max(5 * np.median(residual), 1e-7)
        if refined.sum() < 8 or np.array_equal(good, refined):
            break
        good = refined
    return coef


def _decode(x, coef):
    offset, gain = coef[:4], coef[4:8]
    coupling, interaction, cubic = coef[8:]
    i = np.arange(4)
    j = (i + 1) % 4

    def residual(z, row):
        nxt = z[j]
        return offset + gain*z + coupling*nxt + interaction*z*nxt + cubic*z**3 - row

    def jacobian(z, row):
        g = np.diag(gain + interaction*z[j] + 3*cubic*z*z)
        g[i, j] = coupling + interaction*z
        return g

    z = []
    for row in x:
        result = least_squares(residual, (row-offset)/gain, jac=jacobian,
                               args=(row,), gtol=1e-12, max_nfev=80)
        if np.linalg.norm(result.fun, np.inf) > 1e-8:
            other = least_squares(residual, np.zeros(4), jac=jacobian,
                                  args=(row,), gtol=1e-12, max_nfev=100)
            if np.linalg.norm(other.fun) < np.linalg.norm(result.fun):
                result = other
        z.append(result.x)
    return np.asarray(z).reshape(-1, 4)


def _panel_operators(panel):
    u = np.asarray(panel['u'], dtype=float)
    def f1(x, p):
        return np.c_[_affine(x), np.arctan(p[0]*x), np.sin(p[1]*x)]
    def f2(x, p):
        return _affine(x+p[1]*np.cos(p[0]*x))
    def project(fun, key, start, bounds):
        v = np.asarray(panel[key], dtype=float)
        def residual(p):
            d = fun(u, p)
            return (d @ _fit(d, v) - v).ravel()
        p = least_squares(residual, start, bounds=bounds).x
        bet = _fit(fun(u, p), v)
        return lambda x: fun(x, p) @ bet
    return (project(f1, 'v1', [4.4, 2.58], ([1., 1.], [8., 5.])),
            project(f2, 'v2', [1.28, -.65], ([.6, -1.4], [2.2, -.1])))


def _scalar_design(z, operators):
    a, b = (op(z) for op in operators)
    return np.c_[z, a, b, a*b, a*a, b*b]


def _readout(z, zt, y, operators):
    """Combine a panel-feature prior with smooth main and pairwise effects."""
    features, targets = [_scalar_design(v, operators) for v in (z, zt)]
    center = features.mean(axis=0)
    scale = features.std(axis=0).clip(1e-9)
    h, ht = (features-center)/scale, (targets-center)/scale
    center, scale = z.mean(axis=0), z.std(axis=0).clip(1e-9)
    v, vt = (z-center)/scale, (zt-center)/scale

    def gram(query, query_h):
        k = 1. + query @ v.T + (.75 / h.shape[1]) * (query_h @ h.T)
        delta = np.sqrt(5.) * np.abs(query[:, None, :] - v[None, :, :])
        for length in (1.8, 2.2):
            q = delta/length
            one = (1.+q+q*q/3.) * np.exp(-q)
            pairs = sum(one[..., i]*one[..., j] for i in range(4) for j in range(i+1, 4))
            k += .5 * (one.mean(axis=-1) + pairs/6.)
        return k

    alpha = np.linalg.solve(gram(v, h) + .003*np.eye(len(z)), y)
    return gram(vt, ht) @ alpha


def _trajectory(z, zt, traj, times):
    """Use an affine spatial law and smooth the fitted time coefficients."""
    coef = _fit(_affine(z), traj)
    if len(times) >= 5:
        basis = np.column_stack((times, -np.expm1(-times),
                                 -np.expm1(-2*times), -np.expm1(-3*times)))
        coef = _fit(basis, coef.T).T @ basis.T
    return _affine(zt) @ coef


def solve_batch(study: dict) -> dict:
    train = study['train']
    calibration = _calibration(study['panel_a'])
    z = _decode(np.asarray(train['x'], dtype=float), calibration)
    zt = _decode(np.asarray(study['test']['x'], dtype=float), calibration)
    operators = _panel_operators(study['panel_c'])
    yh = _readout(z, zt, np.asarray(train['y'], dtype=float), operators)
    sp = _trajectory(z, zt, np.asarray(train['traj'], dtype=float),
                     np.asarray(study['eval_times'], dtype=float))
    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])
    return {'y_hat': yh, 'state_pred': sp, 'diagnostics_vec': diag}
