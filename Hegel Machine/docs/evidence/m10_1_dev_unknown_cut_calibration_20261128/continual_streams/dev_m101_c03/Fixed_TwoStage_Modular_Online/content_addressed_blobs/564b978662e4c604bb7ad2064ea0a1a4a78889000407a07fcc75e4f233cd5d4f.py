import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate_a(u, v):
    neighbor = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)
    w = np.ones(len(u))
    for _ in range(16):
        coef = np.stack([_fit(design[:, j, :] * w[:, None], v[:, j] * w)
                         for j in range(u.shape[1])], axis=1)
        residual = np.linalg.norm(np.einsum('ndk,kd->nd', design, coef) - v, axis=1)
        w = 1 / np.hypot(1, residual / 0.02)
    # Refit after removing the three grossly contaminated calibration rows.
    keep = np.argsort(residual)[:-3]
    return np.stack([_fit(design[keep, j, :], v[keep, j])
                     for j in range(u.shape[1])], axis=1)


def _invert_a(x, coef):
    offset, diagonal, coupling, product, cubic = coef
    j = np.arange(x.shape[1])
    k = (j + 1) % len(j)
    matrix = np.diag(diagonal)
    matrix[j, k] = coupling
    u = np.linalg.solve(matrix, (x - offset).T).T

    def residual(a):
        b = a[..., k]
        return offset + diagonal * a + coupling * b + product * a * b + cubic * a**3 - x

    for _ in range(24):
        r = residual(u)
        err = np.linalg.norm(r, axis=1)
        if np.max(err, initial=0) < 1e-11:
            break
        jac = np.zeros((len(u), len(j), len(j)))
        jac[:, j, j] = diagonal + product * u[:, k] + 3 * cubic * u**2
        jac[:, j, k] = coupling + product * u
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        proposal = u - step
        for _ in range(12):
            bad = np.linalg.norm(residual(proposal), axis=1) > err
            if not bad.any():
                break
            proposal[bad] = (u[bad] + proposal[bad]) * 0.5
        u = proposal
    return u


def _panel_c(u, v, query, readout):
    def design(a, scale):
        if readout == 1:
            extra = (a / (1 + scale * np.abs(a)), a * np.abs(a))
        else:
            extra = (np.cos(scale * a) - 1, a**2)
        return np.column_stack((_affine(a), *extra))

    def residual(scale):
        z = design(u, scale[0])
        return (z @ _fit(z, v) - v).ravel()

    scale = least_squares(residual, [0.83 if readout == 1 else 1.3],
                          bounds=(0.25, 2.5)).x[0]
    return design(query, scale) @ _fit(design(u, scale), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    pa, pb, pc = (study[key] for key in ('panel_a', 'panel_b', 'panel_c'))
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    latent = _invert_a(x, _calibrate_a(pa['u'], pa['v']))
    b = _affine(latent) @ _fit(_affine(pb['u']), pb['v'])
    c1 = _panel_c(pc['u'], pc['v1'], latent, 1)
    c2 = _panel_c(pc['u'], pc['v2'], latent, 2)
    return {'n': np.asarray(len(train['x'])),
            'linear': np.column_stack((_affine(b), c1, c2)),
            'mixed': (c1[:, :, None] * c2[:, None, :]).reshape(len(x), -1),
            'y': np.asarray(train['y']), 'traj': np.asarray(train['traj']),
            'eval_times': np.asarray(study['eval_times'])}
