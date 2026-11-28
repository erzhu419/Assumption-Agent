"""Recover latent coordinates from the reference panels and assemble DAG features."""
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


def _readout(b, y):
    """Profile an affine readout, symmetric interaction, and one sine index."""
    pair = b[:, 0]*b[:, 2] + b[:, 1]*b[:, 3]
    base = np.column_stack((_affine(b), pair))
    n = len(y)
    a, x = base[:n], b[:n]
    q, _ = np.linalg.qr(a, mode='reduced')
    centered_y = y - q @ (q.T @ y)

    def residual(h):
        v = np.sin(h[0] + x @ h[1:])
        v = v - q @ (q.T @ v)
        return centered_y - v * ((v @ centered_y) / max(v @ v, 1e-20))

    direction = _fit(a, y)[1:-1]
    starts = [np.r_[phase, mult*direction] for phase in (0., .35, -.35)
              for mult in (.85, 1.55, 2.2, 2.9)]
    best = min((least_squares(residual, h, max_nfev=120) for h in starts),
               key=lambda opt: opt.cost)
    return np.column_stack((base, np.sin(best.x[0] + b @ best.x[1:])))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = (np.asarray(study[k]['x'], float) for k in ('train', 'test'))
    n = len(train)
    z = _inverse(study['panel_a'], np.concatenate((train, test)))
    b = _branch(study['panel_b'], z)
    state = _affine(b)
    readout = _readout(b, np.asarray(study['train']['y'], float))
    return {'z': state[:n], 'zt': state[n:], 'readout_train': readout[:n], 'readout_test': readout[n:]}
