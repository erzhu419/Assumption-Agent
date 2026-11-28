import numpy as np
from scipy.optimize import root


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _robust_fit(design, values, weights):
    for _ in range(30):
        coef = np.linalg.lstsq(
            (design * weights[:, None, None]).reshape(-1, design.shape[-1]),
            (values * weights[:, None]).ravel(), rcond=None,
        )[0]
        residual = np.linalg.norm(design @ coef - values, axis=1)
        weights = 1 / np.hypot(1, residual / max(1e-6, np.median(residual)))
    return coef, residual, weights


def _fit_panel_a(u, v):
    n, d = u.shape
    eye = np.eye(d)
    linear = (_affine(u)[:, None, :, None] * eye[None, :, None, :]).reshape(n, d, -1)
    _, _, weights = _robust_fit(linear, v, np.ones(n))

    # Each channel couples to its cyclic successor with shared linear,
    # quadratic and cubic gains; offsets and diagonal slopes are per channel.
    next_u = np.roll(u, -1, axis=1)
    design = np.concatenate((
        np.broadcast_to(eye, (n, d, d)), eye[None, :, :] * u[:, :, None],
        np.stack((next_u, u * next_u, u ** 3), axis=-1),
    ), axis=-1)
    _, residual, _ = _robust_fit(design, v, weights)
    good = np.argsort(residual)[:-3]
    return np.linalg.lstsq(design[good].reshape(-1, 2*d+3), v[good].ravel(), rcond=None)[0]


def _invert_panel_a(x, coef):
    d = x.shape[1]
    j = np.arange(d)
    nxt = (j + 1) % d
    bias, slope = coef[:d], coef[d:2*d]
    coupling, pair, cubic = coef[-3:]
    initial_jac = np.diag(slope)
    initial_jac[j, nxt] = coupling
    initial = np.linalg.solve(initial_jac, (x - bias).T).T

    def residual(u, observation):
        return bias + slope*u + (coupling + pair*u)*u[nxt] + cubic*u**3 - observation

    def jacobian(u, observation):
        jac = np.diag(slope + pair*u[nxt] + 3*cubic*u*u)
        jac[j, nxt] = coupling + pair*u
        return jac

    return np.array([
        root(residual, guess, args=(row,), jac=jacobian, options={'xtol': 1e-10}).x
        for row, guess in zip(x, initial)
    ]).reshape(x.shape)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    coef_a = _fit_panel_a(a['u'], a['v'])
    coef_b = np.linalg.lstsq(_affine(b['u']), b['v'], rcond=None)[0]
    return {
        key: _affine(_invert_panel_a(study[split]['x'], coef_a)) @ coef_b
        for key, split in (('z_tr', 'train'), ('z_te', 'test'))
    }
