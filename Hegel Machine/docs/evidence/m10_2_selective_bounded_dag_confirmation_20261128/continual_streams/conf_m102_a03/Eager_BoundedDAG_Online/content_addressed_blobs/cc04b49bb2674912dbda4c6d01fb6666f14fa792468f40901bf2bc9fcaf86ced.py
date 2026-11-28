"""Invert the cyclic cubic sensor calibration, then align to readout coordinates."""

import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit_sensor(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    d = u.shape[1]
    nxt = np.roll(u, -1, axis=1)
    # Coordinate-specific offsets and gains; shared cyclic and cubic terms.
    eyes = np.broadcast_to(np.eye(d), (len(u), d, d))
    design = np.concatenate((eyes, u[..., None] * eyes,
                             np.stack((nxt, u * nxt, u ** 3), axis=-1)), axis=-1)
    a = design.reshape(-1, 2 * d + 3)
    b = v.ravel()
    w = np.ones(len(u))
    for _ in range(24):
        rw = np.repeat(w, d)
        coef = np.linalg.lstsq(a * rw[:, None], b * rw, rcond=None)[0]
        err = np.linalg.norm(design @ coef - v, axis=1)
        scale = max(np.median(err), 1e-12)
        w = np.sqrt(np.minimum(1., 1.5 * scale / np.maximum(err, 1e-12)))

    good = err <= 5. * scale
    if np.count_nonzero(good) * d >= a.shape[1]:
        refit, _, rank, _ = np.linalg.lstsq(design[good].reshape(-1, a.shape[1]),
                                           v[good].ravel(), rcond=None)
        if rank == a.shape[1]:
            coef = refit
    return coef[:d], coef[d:2*d], coef[2*d:]


def _sensor(u, offset, gain, coupling, interaction, cubic):
    nxt = np.roll(u, -1, axis=1)
    return offset + gain * u + (coupling + interaction * u) * nxt + cubic * u**3


def _inverse(x, calibration):
    offset, gain, (coupling, interaction, cubic) = calibration
    j = np.arange(x.shape[1])
    nxt = (j + 1) % len(j)
    linear = np.diag(gain)
    linear[j, nxt] += coupling
    u = np.linalg.solve(linear, (x - offset).T).T
    for _ in range(40):
        res = _sensor(u, offset, gain, coupling, interaction, cubic) - x
        error = np.linalg.norm(res, axis=1)
        if not len(error) or np.max(error) <= 1e-11:
            break
        jac = np.broadcast_to(linear, (len(u), *linear.shape)).copy()
        jac[:, j, j] += 3. * cubic * u**2 + interaction * u[:, nxt]
        jac[:, j, nxt] += interaction * u
        step = np.linalg.solve(jac, res[..., None])[..., 0]
        rate = np.ones((len(u), 1))
        for _ in range(18):
            trial = u - rate * step
            trial_err = np.linalg.norm(_sensor(trial, offset, gain, coupling, interaction, cubic) - x, axis=1)
            if np.all((trial_err < error) | (error <= 1e-11)):
                break
            rate[trial_err >= error] *= 0.5
        u = np.where((trial_err < error)[:, None], trial, u)
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    calibration = _fit_sensor(study['panel_a'])
    panel = study['panel_b']
    align = np.linalg.lstsq(_affine(np.asarray(panel['u'], dtype=float)),
                            np.asarray(panel['v'], dtype=float), rcond=None)[0]

    def encode(rows):
        return _affine(_inverse(np.asarray(rows, dtype=float), calibration)) @ align

    return {'z': encode(study['train']['x']), 'zt': encode(study['test']['x'])}
