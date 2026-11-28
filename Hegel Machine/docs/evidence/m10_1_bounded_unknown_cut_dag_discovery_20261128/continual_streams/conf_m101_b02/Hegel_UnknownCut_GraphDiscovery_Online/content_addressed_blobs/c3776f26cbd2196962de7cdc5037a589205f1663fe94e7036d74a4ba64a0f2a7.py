import numpy as np


def _features(u, j):
    a, b = u[:, j], u[:, (j + 1) % u.shape[1]]
    return np.column_stack((np.ones(len(u)), a, b, a * b, a ** 3))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    u, v = (study['panel_a'][key] for key in ('u', 'v'))
    d = u.shape[1]
    designs = [_features(u, j) for j in range(d)]
    weights = np.ones(len(u))
    for _ in range(24):
        coef = np.array([np.linalg.lstsq(X * weights[:, None], v[:, j] * weights, rcond=None)[0]
                         for j, X in enumerate(designs)])
        residual = np.linalg.norm(np.column_stack([X @ c for X, c in zip(designs, coef)]) - v, axis=1)
        scale = max(1e-8, 1.5 * np.median(residual))
        weights = 1 / np.hypot(1, residual / scale)

    observed = np.vstack((study['train']['x'], study['test']['x']))
    z = observed.copy()
    offset, linear, cross, product, cubic = coef.T
    j = np.arange(d)
    nxt = (j + 1) % d
    for _ in range(24):
        b = z[:, nxt]
        r = offset + linear * z + cross * b + product * z * b + cubic * z ** 3 - observed
        if np.max(np.abs(r)) < 1e-10:
            break
        jac = np.zeros((len(z), d, d))
        jac[:, j, j] = linear + product * b + 3 * cubic * z ** 2
        jac[:, j, nxt] = cross + product * z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= step / np.maximum(1, np.max(np.abs(step), axis=1, keepdims=True))
    return {'z': z}
