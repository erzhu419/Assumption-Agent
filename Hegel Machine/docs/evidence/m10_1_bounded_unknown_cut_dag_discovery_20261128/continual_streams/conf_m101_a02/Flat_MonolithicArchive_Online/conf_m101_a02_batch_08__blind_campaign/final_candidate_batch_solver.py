"""Fit panel operators and predict scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    rough = _aff(u)
    keep = np.argsort(np.linalg.norm(v - rough @ _fit(rough, v), axis=1))[:-3]
    def design(z, j):
        p, q = z[:, j], z[:, (j+1)%4]
        return np.column_stack((np.ones(len(z)), p, q, p*q, p**3))
    coef = np.column_stack([_fit(design(u[keep], j), v[keep, j]) for j in range(4)])
    rough = _fit(rough[keep], v[keep])
    z = np.linalg.solve(rough[1:].T, (x-rough[0]).T).T
    for _ in range(10):
        p, q = z, np.roll(z, -1, axis=1)
        val = coef[0] + p*coef[1] + q*coef[2] + p*q*coef[3] + p**3*coef[4]
        jac = np.zeros((len(z), 4, 4))
        j = np.arange(4)
        jac[:, j, j] = coef[1] + q*coef[3] + 3*p*p*coef[4]
        jac[:, j, (j+1)%4] = coef[2] + p*coef[3]
        z -= np.linalg.solve(jac, (val-x)[..., None])[..., 0]
    return z


def _panel(p, key, z, basis, init, bounds):
    u, v = p['u'], p[key]
    def residual(theta):
        a = basis(u, theta)
        return (a @ _fit(a, v) - v).ravel()
    theta = least_squares(residual, init, bounds=bounds, ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return basis(z, theta) @ _fit(basis(u, theta), v)


def _coords(study):
    z = _calibrate(study['panel_a'], np.concatenate((study['train']['x'], study['test']['x'])))
    b = _panel(study['panel_b'], 'v', z, lambda x,p: _aff(x*(1-p[1]*np.exp(-p[0]*x*x))), [1,1.1], ([.1,.05],[5,4]))
    p = study['panel_c']
    c = _panel(p, 'v1', z, lambda x,p: np.c_[_aff(x),np.tanh(p[0]*x)], [2], (.2,5))
    d = _panel(p, 'v2', z, lambda x,p: np.c_[_aff(x),np.cos(p*x)], [1.27]*4, (.2,4))
    return z, b, c, d


def _blocks(study):
    z, b, c, d = _coords(study)
    a = np.sum(z * np.roll(b, 1, axis=1), axis=1)
    opp = np.sum(b * np.roll(b, 2, axis=1), axis=1)
    return z, b, c, d, a, opp


def _scalar_designs(x, z, b, c, d, a, opp):
    products = z * np.roll(b, 1, axis=1)
    auxiliary = np.sum(c * np.roll(products, 3, axis=1), axis=1)
    cubic = np.sum(b * products, axis=1)
    common = np.c_[_aff(b), d, a, opp, auxiliary, cubic]
    energy = np.sum(d**2, axis=1)
    rational = np.c_[common, energy, opp / (1 + 0.2 * energy**2)]
    polynomial = np.c_[common, np.sum((d * x)**2, axis=1),
                       b[:, 3] * np.exp(-1.7 * b[:, 3]**2), z[:, 2]**3]
    return rational, polynomial


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['y'])
    z, b, c, d, a, opp = _blocks(study)
    x = np.concatenate((train['x'], study['test']['x']))
    rational, polynomial = _scalar_designs(x, z, b, c, d, a, opp)
    # Blend the compact readouts: the rational fit improves ordinary holdouts,
    # while the polynomial fit provides a better high-amplitude extrapolation.
    y = (0.7 * (rational[n:] @ _fit(rational[:n], train['y']))
         + 0.3 * (polynomial[n:] @ _fit(polynomial[:n], train['y'])))
    g = np.c_[_aff(b), z, a]
    state = g[n:] @ _fit(g[:n], train['traj'])
    terminal = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
