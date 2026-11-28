import numpy as np


def _fit_forward(u, v):
    neighbor = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)

    def fit(weights):
        return np.array([
            np.linalg.lstsq(design[:, j] * weights[:, None], v[:, j] * weights, rcond=None)[0]
            for j in range(u.shape[1])
        ]).T

    coef = fit(np.ones(len(u)))
    for _ in range(30):
        residual = np.linalg.norm(v - np.einsum('njk,kj->nj', design, coef), axis=1)
        scale = max(4e-4, 1.5 * np.median(residual))
        coef = fit(1.0 / np.hypot(1.0, residual / scale))
    residual = np.linalg.norm(v - np.einsum('njk,kj->nj', design, coef), axis=1)
    keep = np.argsort(residual)[:-3]
    weights = np.zeros(len(u))
    weights[keep] = 1
    return fit(weights)


def _invert(v, coef):
    bias, linear, coupling, interaction, cubic = coef
    j = np.arange(len(bias))
    k = (j + 1) % len(bias)
    affine = np.diag(linear)
    affine[j, k] = coupling
    u = np.linalg.solve(affine, (v - bias).T).T.copy()
    jacobian = np.broadcast_to(affine, (len(u), *affine.shape)).copy()
    for _ in range(25):
        neighbor = u[:, k]
        residual = bias + linear * u + coupling * neighbor + interaction * u * neighbor + cubic * u**3 - v
        if np.max(np.abs(residual)) < 1e-11:
            break
        jacobian[:, j, j] = linear + interaction * neighbor + 3 * cubic * u**2
        jacobian[:, j, k] = coupling + interaction * u
        u -= np.linalg.solve(jacobian, residual[..., None])[..., 0]
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    coef = _fit_forward(a['u'], a['v'])
    return {'u_tr': _invert(study['train']['x'], coef),
            'u_te': _invert(study['test']['x'], coef),
            'b_u': b['u'], 'b_v': b['v']}
