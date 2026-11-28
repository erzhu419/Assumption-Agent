import numpy as np


_NEXT = np.array([1, 2, 3, 0])
_AXIS = np.arange(4)


def _fit_forward(u, v):
    # Per-channel offset and gain; shared cyclic interaction and diagonal cubic.
    n = len(u)
    design = np.zeros((n, 4, 10))
    design[:, _AXIS, _AXIS] = 1.0
    design[:, _AXIS, 4 + _AXIS] = u
    design[:, :, 8] = (u - 1.0) * u[:, _NEXT]
    design[:, :, 9] = u**3

    weights = np.ones(n)
    for _ in range(35):
        d = design * weights[:, None, None]
        beta = np.linalg.lstsq(d.reshape(-1, 10), (v * weights[:, None]).ravel(), rcond=None)[0]
        error = np.linalg.norm(design @ beta - v, axis=1)
        cutoff = max(1.5 * np.median(error), 1e-12)
        weights = np.sqrt(cutoff / np.maximum(error, cutoff))
    keep = np.argsort(error)[:-3]
    return np.linalg.lstsq(design[keep].reshape(-1, 10), v[keep].ravel(), rcond=None)[0]


def _inverse(x, beta):
    offset, gain = beta[:4], beta[4:8]
    interaction, cubic = beta[8:]
    linear = np.diag(gain)
    linear[_AXIS, _NEXT] = -interaction
    z = np.linalg.solve(linear, (x - offset).T).T

    def residual(q):
        return offset + gain * q + interaction * (q - 1.0) * q[:, _NEXT] + cubic * q**3 - x

    for _ in range(30):
        error = residual(z)
        norm = np.max(np.abs(error), axis=1)
        if np.max(norm) < 1e-12:
            break
        jac = np.zeros((len(z), 4, 4))
        jac[:, _AXIS, _AXIS] = gain + interaction * z[:, _NEXT] + 3.0 * cubic * z**2
        jac[:, _AXIS, _NEXT] = interaction * (z - 1.0)
        step = np.linalg.solve(jac, error[..., None])[..., 0]
        trial = z - step
        for _ in range(12):
            worse = np.max(np.abs(residual(trial)), axis=1) > norm
            if not np.any(worse):
                break
            trial[worse] = 0.5 * (trial[worse] + z[worse])
        z = trial
    return z


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel = study['panel_a']
    beta = _fit_forward(panel['u'], panel['v'])
    return {key: _inverse(study[split]['x'], beta)
            for key, split in (('z_tr', 'train'), ('z_te', 'test'))}
