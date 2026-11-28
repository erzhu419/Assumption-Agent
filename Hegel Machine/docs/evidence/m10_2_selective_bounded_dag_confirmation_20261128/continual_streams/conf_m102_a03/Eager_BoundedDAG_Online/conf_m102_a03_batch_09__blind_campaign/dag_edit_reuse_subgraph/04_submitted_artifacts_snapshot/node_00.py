"""Recover latent coordinates from paired sensor calibration measurements."""
import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    affine = _affine(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(4):
        residual = np.linalg.norm(affine @ _fit(affine[keep], v[keep]) - v, axis=1)
        next_keep = residual < 4 * max(np.median(residual), 1e-9)
        if next_keep.sum() < max(12, len(u) // 2) or np.array_equal(keep, next_keep):
            break
        keep = next_keep
    u, v = u[keep], v[keep]
    d = u.shape[1]
    opposite = np.roll(u, -(d // 2), axis=1)
    eye = np.broadcast_to(np.eye(d), (len(u), d, d)).reshape(-1, d)
    base = np.column_stack((eye, eye * u.reshape(-1, 1)))
    y = v.ravel()

    def profile(freq):
        design = np.column_stack((base, np.tanh(freq[0] * u).ravel(),
                                  np.sin(freq[1] * opposite).ravel()))
        coef = _fit(design, y)
        return design @ coef - y, coef

    freq = np.array([1.8, 2.4])
    for _ in range(12):
        residual, coef = profile(freq)
        h = 1e-4
        jac = np.column_stack([(profile(freq + h * e)[0] - profile(freq - h * e)[0]) / (2 * h)
                               for e in np.eye(2)])
        step = _fit(jac, -residual)
        if np.max(np.abs(step)) < 1e-8:
            break
        loss = residual @ residual
        for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = np.clip(freq + shrink * step, [1.0, 1.7], [2.8, 3.1])
            err, _ = profile(trial)
            if err @ err < loss:
                freq = trial
                break
        else:
            break
    _, coef = profile(freq)
    return coef[:d], coef[d:2*d], coef[-2], coef[-1], freq


def _invert(rows, model):
    bias, gain, soft, coupling, freq = model
    shift = -(len(gain) // 2)
    target = rows - bias
    x = target / gain
    upper_slope = gain + abs(soft * freq[0])
    for _ in range(360):
        err = target - gain * x - soft * np.tanh(freq[0] * x)
        err -= coupling * np.sin(freq[1] * np.roll(x, shift, axis=1))
        if not err.size or np.max(np.abs(err)) < 1e-11:
            break
        x += err / upper_slope
    return x


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    model = _calibrate(study['panel_a'])
    panel = study['panel_b']
    transform = _fit(_affine(np.asarray(panel['u'], dtype=float)), np.asarray(panel['v'], dtype=float))

    def encode(rows):
        return _affine(_invert(np.asarray(rows, dtype=float), model)) @ transform

    return {'z': encode(study['train']['x']), 'zt': encode(study['test']['x'])}
