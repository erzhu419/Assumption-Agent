"""Recover latent coordinates from paired calibration measurements."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _calibrate(panel, measured):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.linalg.norm(v-u, axis=1) < 3
    u, v = u[keep], v[keep]
    opposite = np.roll(u, 2, axis=1)

    def profile(freq, coefficients=False):
        cols = [np.column_stack((np.ones(len(u)), u[:, j],
                                 np.sin(freq[0]*opposite[:, j]),
                                 np.sin(freq[1]*u[:, j]))) for j in range(u.shape[1])]
        w = np.column_stack([np.linalg.lstsq(d, v[:, j], rcond=None)[0]
                             for j, d in enumerate(cols)])
        if coefficients:
            return w
        return np.concatenate([d @ w[:, j] - v[:, j] for j, d in enumerate(cols)])

    freq = least_squares(profile, [2.4, 3.1], bounds=([1., 1.], [5., 6.])).x
    offset, gain, coupled, own = profile(freq, True)
    x = (measured-offset)/gain
    for _ in range(120):
        update = (measured-offset-coupled*np.sin(freq[0]*np.roll(x, 2, axis=1))
                  - own*np.sin(freq[1]*x))/gain
        if np.max(np.abs(update-x)) < 1.e-11:
            break
        x = update
    return update


def _profile_map(panel, key, query, basis, bounds):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', key))

    def fit(k):
        d = basis(u, k)
        w = np.linalg.lstsq(d, v, rcond=None)[0]
        return np.sum((d@w-v)**2), w

    grid = np.linspace(*bounds, 41)
    errors = [fit(k)[0] for k in grid]
    j = int(np.argmin(errors))
    best = minimize_scalar(lambda k: fit(k)[0], bounds=(grid[max(0,j-1)], grid[min(len(grid)-1,j+1)]),
                           method='bounded', options={'xatol': 1.e-10}).x
    return basis(query, best) @ fit(best)[1]


def _b_basis(x, k):
    return np.column_stack((np.ones(len(x)), np.cos(k*x)))


def _c_map(panel, query):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1'))

    def design(x, freq):
        return np.column_stack((np.ones(len(x)), np.sin(freq[0]*x), np.tanh(freq[1]*x)))

    def profile(freq, coefficients=False):
        d = design(u, freq)
        w = np.linalg.lstsq(d, v, rcond=None)[0]
        return w if coefficients else (d @ w - v).ravel()

    freq = least_squares(profile, [2.1, 1.75], bounds=([1., .6], [3.5, 3.5])).x
    return design(query, freq) @ profile(freq, True)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']['x']
    x = _calibrate(study['panel_a'], np.concatenate((train, study['test']['x'])))
    b = _profile_map(study['panel_b'], 'v', x, _b_basis, (0.9, 5.5))
    c = _c_map(study['panel_c'], b)
    n = len(train)
    return dict(x_train=x[:n], x_test=x[n:], c_train=c[:n], c_test=c[n:])
