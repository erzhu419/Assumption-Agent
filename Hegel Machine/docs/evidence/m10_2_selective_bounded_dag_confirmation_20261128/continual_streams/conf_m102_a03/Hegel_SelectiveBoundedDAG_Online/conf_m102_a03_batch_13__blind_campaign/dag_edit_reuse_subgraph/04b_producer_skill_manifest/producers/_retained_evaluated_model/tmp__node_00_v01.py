"""Recover latent coordinates from the reference panels and assemble DAG features."""
import itertools
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def _affine(a):
    return np.column_stack((np.ones(len(a)), a))


def _sensor_basis(u, j):
    a, b = u[:, j], u[:, (j + 1) % u.shape[1]]
    return np.column_stack((np.ones(len(a)), a, b, a*b, a**3))


def _inverse(panel, x):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    coeff = []
    for j in range(u.shape[1]):
        a = _sensor_basis(u, j)
        y = v[:, j]
        c = least_squares(lambda c: a @ c - y, _fit(a, y), loss='soft_l1', f_scale=.003).x
        r = np.abs(a @ c - y)
        good = r <= max(.003, 5*np.median(r))
        coeff.append(_fit(a[good], y[good]))
    coeff = np.asarray(coeff)
    z = np.linalg.solve(np.diag(coeff[:, 1]) + np.roll(np.diag(coeff[:, 2]), 1, axis=1), (x-coeff[:, 0]).T).T
    for _ in range(15):
        nxt = np.roll(z, -1, axis=1)
        f = coeff[:, 0] + coeff[:, 1]*z + coeff[:, 2]*nxt + coeff[:, 3]*z*nxt + coeff[:, 4]*z**3 - x
        jac = np.zeros((len(z), z.shape[1], z.shape[1]))
        ii = np.arange(z.shape[1])
        jac[:, ii, ii] = coeff[:, 1] + coeff[:, 3]*nxt + 3*coeff[:, 4]*z*z
        jac[:, ii, (ii+1) % z.shape[1]] = coeff[:, 2] + coeff[:, 3]*z
        step = np.linalg.solve(jac, f[..., None])[..., 0]
        z -= step
        if np.max(np.abs(step), initial=0.) < 1e-12:
            break
    return z


def _branch(panel, z):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    def basis(a, h):
        return np.column_stack((_affine(a), np.cos(h*a)))
    def err(h):
        a = basis(u, h)
        return np.sum((a @ _fit(a, v) - v)**2)
    grid = np.linspace(.2, 8., 80)
    j = int(np.argmin([err(h) for h in grid]))
    h = minimize_scalar(err, bounds=(grid[max(0,j-1)], grid[min(len(grid)-1,j+1)]), method='bounded', options={'xatol': 1e-10}).x
    return basis(z, h) @ _fit(basis(u, h), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = (np.asarray(study[k]['x'], float) for k in ('train', 'test'))
    n = len(train)
    z = _inverse(study['panel_a'], np.concatenate((train, test)))
    b = _branch(study['panel_b'], z)
    state = _affine(b)
    terms = [np.prod(b[:, p], axis=1) for k in (2, 3) for p in itertools.combinations_with_replacement(range(b.shape[1]), k)]
    readout = np.column_stack((state, *terms, z))
    return {'z': state[:n], 'zt': state[n:], 'readout_train': readout[:n], 'readout_test': readout[n:]}
