"""Fit panel operators and predict scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.linalg import solve


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


def _grams(coords, n):
    out = []
    for x in coords:
        a = (x-x[:n].mean(0))/x[:n].std(0)
        out.append(a @ a.T / a.shape[1])
    return out


def _predict(k, y, reg):
    n = len(y)
    return k[n:, :n] @ solve(k[:n,:n]+reg*np.eye(n), y, assume_a='pos')


def _shared_amplitude(times, trajectories, coordinates):
    # Profile the relaxation rate after removing the linear panel responses.
    q = np.linalg.qr(coordinates, mode='reduced')[0]
    residual = trajectories - q @ (q.T @ trajectories)

    def mode(rate):
        return -np.expm1(-rate * times)

    def loss(rate):
        h = mode(rate)
        return np.linalg.norm(residual - np.outer(residual @ h / (h @ h), h))

    h = mode(minimize_scalar(loss, bounds=(.1, 4), method='bounded').x)
    return trajectories @ h / (h @ h)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['y'])
    z, b, c, d = _coords(study)
    Z, B, C, D = _grams((z, b, c, d), n)
    linear = np.c_[_aff(z[:n]), b[:n], c[:n], d[:n]]
    amplitude = _shared_amplitude(study['eval_times'], tr['traj'], linear)

    # Fit the common state contribution and a smoother scalar correction.
    k_state = 1 + B + .3*B**2 + .3*(Z+C+D) + 2*Z*B
    k_amplitude = 1 + B + .3*B**2 + Z + 2*Z*B
    k_contrast = 1 + Z + B + .25*(C+D) + .03*B**2
    y = (_predict(k_amplitude, amplitude, .0001)
         + _predict(k_contrast, tr['y'] - amplitude, .06))
    state = _predict(k_state, tr['traj'], .001)
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
