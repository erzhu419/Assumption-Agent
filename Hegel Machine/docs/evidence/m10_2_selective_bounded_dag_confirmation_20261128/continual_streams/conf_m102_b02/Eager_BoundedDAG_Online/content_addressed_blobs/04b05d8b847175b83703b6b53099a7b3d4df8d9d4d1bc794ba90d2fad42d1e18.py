"""Recover the shared coordinates from independent calibration panels."""
import numpy as np


def _affine_features(u):
    return np.column_stack((np.ones(len(u)), u))


def _panel_a_features(u, frequency):
    return np.column_stack((_affine_features(u), np.sin(frequency * u)))


def _fit_panel_a(u, v, frequency):
    affine = _affine_features(u)
    initial = np.linalg.lstsq(affine, v, rcond=None)[0]
    residual = np.linalg.norm(affine @ initial - v, axis=1)
    keep = residual < max(0.002, 4.0 * np.median(residual))
    features = _panel_a_features(u, frequency)
    for _ in range(8):
        coef = np.linalg.lstsq(features[keep], v[keep], rcond=None)[0]
        residual = np.linalg.norm(features @ coef - v, axis=1)
        update = residual < max(0.002, 4.0 * np.median(residual))
        if update.sum() < features.shape[1] or np.array_equal(update, keep):
            break
        keep = update
    return coef, np.linalg.lstsq(affine[keep], v[keep], rcond=None)[0]


def _invert_panel_a(x, coef, affine, frequency):
    d = x.shape[1]
    u = np.linalg.solve(affine[1:].T, (x - affine[0]).T).T
    for _ in range(25):
        residual = _panel_a_features(u, frequency) @ coef - x
        error = np.linalg.norm(residual, axis=1)
        if np.max(error, initial=0.0) < 1e-12:
            break
        jac = coef[None, 1:d+1, :] + (frequency * np.cos(frequency * u))[..., None] * coef[None, d+1:, :]
        step = np.linalg.solve(jac.transpose(0, 2, 1), residual[..., None])[..., 0]
        step /= np.maximum(1.0, np.max(np.abs(step), axis=1, keepdims=True) / 0.5)
        for _ in range(12):
            trial = u - step
            improve = np.linalg.norm(_panel_a_features(trial, frequency) @ coef - x, axis=1) < error
            if np.all(improve | (error < 1e-12)):
                break
            step[~improve] *= 0.5
        u = np.where(improve[:, None], trial, u)
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    a = study['panel_a']
    frequency = float(study['telemetry_vec'][1])
    coef, affine = _fit_panel_a(np.asarray(a['u'], float), np.asarray(a['v'], float), frequency)
    u = _invert_panel_a(np.vstack((train, test)), coef, affine, frequency)
    b = study['panel_b']
    mapping = np.linalg.lstsq(_affine_features(b['u']), b['v'], rcond=None)[0]
    coordinates = _affine_features(u) @ mapping
    return {'z': coordinates[:len(train)].copy(), 'zt': coordinates[len(train):].copy()}
