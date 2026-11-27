"""Robust forward calibration and inversion of the four-channel rosette."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares

_TERMS = tuple(np.array(list(combinations_with_replacement(range(4), d)))
               for d in (2, 3))
_EYE = np.eye(4)


def _basis(x):
    return np.column_stack((np.ones(len(x)), x,
                            *(x[:, t].prod(axis=2) for t in _TERMS)))


def _row_size(r):
    return np.sqrt(np.mean(r * r, axis=1)) if r.ndim == 2 else np.abs(r)


def _robust_linear(a, y):
    """Fit all outputs with a shared weight for each sweep row."""
    w = np.ones(len(a))
    for _ in range(14):
        wa = a * w[:, None]
        wy = y * (w[:, None] if y.ndim == 2 else w)
        c = np.linalg.lstsq(wa, wy, rcond=1e-10)[0]
        r = _row_size(a @ c - y)
        scale = max(2e-5, 2.8 * np.median(r))
        updated = 1 / np.hypot(1, r / scale)
        if np.max(np.abs(w - updated)) < 0.003:
            break
        w = updated
    m = r <= max(8e-5, 7 * np.median(r))
    if m.sum() >= a.shape[1] + 3 and not np.all(m):
        c = np.linalg.lstsq(a[m], y[m], rcond=1e-10)[0]
    return c


def _clean_rows(a, y):
    """Screen EMI jointly using a low-leverage fit before fitting richer models."""
    quadratic = a[:, :15]
    residual = _row_size(quadratic @ _robust_linear(quadratic, y) - y)
    return residual <= max(1e-4, 5 * np.median(residual))


def _rational_predict(x, a, q, c):
    return a[:, :5] @ c[:5] + (a[:, 5:15] @ c[5:]) / (1 + (x @ q)**2)


def _rational_fit(x, a, y):
    """Profile the numerator and affine terms at each saturation direction."""
    r = np.abs(a @ _robust_linear(a, y) - y)
    # Remove clearly separated gross spikes, not the residual curvature.
    keep = r <= max(1e-4, 9 * np.median(r))
    if keep.sum() < 22:
        keep[:] = True
    x, a, y = x[keep], a[keep], y[keep]

    def profile(q, w):
        mat = np.column_stack((a[:, :5], a[:, 5:] / (1 + (x @ q)**2)[:, None]))
        mat = mat * w[:, None]
        b = y * w
        c = np.linalg.lstsq(mat, b, rcond=1e-10)[0]
        return mat @ c - b, c

    def fit(start, w):
        return least_squares(lambda q: profile(q, w)[0], start, bounds=(-6., 6.),
                             max_nfev=55, gtol=1e-11, ftol=1e-8, xtol=1e-8)

    w = np.ones(len(x))
    starts = list(.75 * _EYE)
    starts += [.55 * (_EYE[i] + sign * _EYE[j]) for i in range(4)
               for j in range(i + 1, 4) for sign in (-1, 1)]
    starts.sort(key=lambda q: np.linalg.norm(profile(q, w)[0]))
    q = min((fit(q0, w) for q0 in starts[:4]), key=lambda opt: opt.cost).x
    for _ in range(5):
        r, c = profile(q, w)
        r = np.abs(r / w)
        updated = 1 / np.hypot(1, r / max(2e-5, 2.8 * np.median(r)))
        if np.max(np.abs(w - updated)) < .01:
            break
        w = updated
        q = fit(q, w).x
    r, c = profile(q, w)
    r = np.abs(r / w)
    inlier = r <= max(8e-5, 7 * np.median(r))
    if inlier.sum() >= 22 and not np.all(inlier):
        w = inlier.astype(float)
        q = fit(q, w).x
        c = profile(q, w)[1]
    return q, c


def _score(err):
    """Trim gross sweep spikes from out-of-fold comparison, not from inputs."""
    r = np.sort(np.abs(err), axis=0)
    return np.sqrt(np.mean(r[:max(1, len(r) * 85 // 100)]**2, axis=0))


def _fit_mapping(x, y):
    a = _basis(x)
    affine = _robust_linear(a[:, :5], y)
    if np.quantile(_row_size(a[:, :5] @ affine - y), .80) < 3e-5:
        return affine, [None] * 4

    folds = np.arange(len(x)) % 4
    sizes = [p for p in (5, 15, 35) if p + 8 < len(x) * .75]
    if len(sizes) < 2:
        return affine, [None] * 4
    cv = np.empty((len(sizes), *y.shape))
    for k in range(4):
        train = folds != k
        held = ~train
        train[train] = _clean_rows(a[train], y[train])
        for i, p in enumerate(sizes):
            c = _robust_linear(a[train, :p], y[train])
            cv[i, held] = a[held, :p] @ c - y[held]
    scores = np.array([_score(e) for e in cv])
    chosen = np.argmax(scores <= 1.15 * scores.min(axis=0) + 4e-6, axis=0)
    poly = np.zeros((max(sizes[i] for i in chosen), 4))
    clean = _clean_rows(a, y)
    for i in np.unique(chosen):
        p = sizes[i]
        poly[:p, chosen == i] = _robust_linear(a[clean, :p], y[clean][:, chosen == i])

    rational = [None] * 4
    for j, i in enumerate(chosen):
        if scores[i, j] < 3e-5 or len(x) < 40:
            continue
        err = np.empty(len(x))
        for k in range(4):
            train = folds != k
            held = ~train
            train[train] = _clean_rows(a[train], y[train])
            model = _rational_fit(x[train], a[train, :15], y[train, j])
            err[held] = _rational_predict(x[held], a[held], *model) - y[held, j]
        if _score(err) < .8 * scores[i, j]:
            rational[j] = _rational_fit(x[clean], a[clean, :15], y[clean, j])
    return poly, rational


def calibrate_upstream(study: dict) -> dict:
    """Return a study with readings and calibration sweeps in reference coordinates."""
    sweeps = study['sensor_calibration_sweeps']
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    poly, rational = _fit_mapping(x, y)
    affine = poly[:5].copy()
    for j, model in enumerate(rational):
        if model is not None:
            affine[:, j] = model[1][:5]

    def forward(z):
        a = _basis(z)
        result = a[:, :len(poly)] @ poly
        for j, model in enumerate(rational):
            if model is not None:
                result[:, j] = _rational_predict(z, a, *model)
        return result

    def inverse(raw):
        array = np.asarray(raw, dtype=float)
        readings = array.reshape(-1, 4)
        initial = np.linalg.solve(affine[1:].T, (readings - affine[0]).T).T
        if len(poly) == 5 and not any(m is not None for m in rational):
            return initial.reshape(array.shape)
        for i, goal in enumerate(readings):
            initial[i] = least_squares(lambda z: forward(z[None])[0] - goal, initial[i],
                                       gtol=1e-12, max_nfev=70).x
        return initial.reshape(array.shape)

    calibrated = dict(study, y_obs_raw=inverse(study['y_obs_raw']))
    bank = sweeps['single_bank_sweeps']
    calibrated['sensor_calibration_sweeps'] = dict(
        sweeps, raw_rosette=sweeps['ref_rosette'].copy(),
        single_bank_sweeps=dict(bank, raw_readout=bank['ref_signal'].copy()))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(p, y_pulse_raw=inverse(p['y_pulse_raw'])) if p['y_pulse_raw'] is not None
            else dict(p) for p in pulses]
    return calibrated
