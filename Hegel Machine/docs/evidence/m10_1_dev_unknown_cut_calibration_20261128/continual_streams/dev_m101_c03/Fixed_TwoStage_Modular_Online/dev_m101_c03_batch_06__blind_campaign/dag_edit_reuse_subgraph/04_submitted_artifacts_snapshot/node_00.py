import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _panel_a_inverse(panel, observed):
    u, v = panel['u'], panel['v']
    i = np.arange(u.shape[1])
    j = (i + 1) % len(i)
    next_u = u[:, j]
    design = np.stack((np.ones_like(u), u, next_u, u * next_u, u**3), axis=2)
    weights = np.ones(len(u))
    for _ in range(25):
        coef = np.column_stack([_fit(design[:, k] * weights[:, None], v[:, k] * weights) for k in i])
        errors = np.linalg.norm(np.einsum('nik,ki->ni', design, coef) - v, axis=1)
        scale = max(1e-6, 1.5 * np.median(errors))
        weights = 1 / np.sqrt(1 + (errors / scale)**2)

    offset, gain, coupling, cross, cubic = coef

    def forward(z):
        return offset + gain * z + coupling * z[j] + cross * z * z[j] + cubic * z**3

    def jacobian(z):
        jac = np.diag(gain + cross * z[j] + 3 * cubic * z**2)
        jac[i, j] += coupling + cross * z
        return jac

    return np.array([least_squares(lambda z: forward(z) - row, row - offset,
                                  jac=jacobian, gtol=1e-11).x for row in observed])


def _panel_c_features(panel, u):
    def pair_design(z):
        return _intercept(z * np.roll(z, -1, axis=1))

    def cosine_design(z, frequency):
        return _intercept(np.cos(frequency * z))

    ref = panel['u']
    c = pair_design(u) @ _fit(pair_design(ref), panel['v1'])

    def cosine_loss(frequency):
        a = cosine_design(ref, frequency)
        return np.sum((a @ _fit(a, panel['v2']) - panel['v2'])**2)

    frequency = minimize_scalar(cosine_loss, bounds=(1.4, 3.4), method='bounded').x
    d = cosine_design(u, frequency) @ _fit(cosine_design(ref, frequency), panel['v2'])
    return c, d


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    u = _panel_a_inverse(study['panel_a'], np.concatenate((train['x'], test['x'])))
    c, d = _panel_c_features(study['panel_c'], u)
    mixed = (c[:, :, None] * d[:, None, :]).reshape(len(u), -1)
    prev_u = np.roll(u, 1, axis=1)
    next_u = np.roll(u, -1, axis=1)
    cyclic = np.column_stack((u**3, u**2 * prev_u, u**2 * next_u, prev_u * u * next_u))
    linear = np.column_stack((_intercept(u), c, d, cyclic, mixed))
    return dict(n=len(train['x']), linear=linear, mixed=mixed,
                eval_times=study['eval_times'], y=train['y'], traj=train['traj'])
