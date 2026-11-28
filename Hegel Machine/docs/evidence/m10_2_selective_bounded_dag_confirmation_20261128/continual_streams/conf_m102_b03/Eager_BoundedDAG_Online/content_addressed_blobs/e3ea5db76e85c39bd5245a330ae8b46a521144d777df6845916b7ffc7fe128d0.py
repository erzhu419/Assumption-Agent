"""Recover calibrated coordinates and predict the two branch readouts."""
import numpy as np
from scipy.optimize import least_squares, root


def _ls(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _intercept(features):
    return np.column_stack((np.ones(len(features)), features))


def _calibration_features(z, neighbor):
    return _intercept(np.column_stack((z, neighbor, z * neighbor, z**3)))


def _calibrate(panel, observed):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    width = u.shape[1]
    right = (np.arange(width) + 1) % width
    coefficients = []
    for j, k in enumerate(right):
        design = _calibration_features(u[:, j], u[:, k])
        y = v[:, j]
        robust = least_squares(lambda p: design @ p - y, _ls(design, y),
                               loss='soft_l1', f_scale=.02).x
        error = design @ robust - y
        center = np.median(error)
        mad = np.median(np.abs(error - center))
        clean = np.abs(error - center) <= max(.025, 8 * mad)
        coefficients.append(_ls(design[clean], y[clean]) if clean.sum() >= 5 else robust)
    offset, gain, coupling, product, cubic = np.asarray(coefficients).T
    linear = np.diag(gain)
    linear[np.arange(width), right] = coupling
    starts = np.linalg.solve(linear, (observed - offset).T).T
    result = []
    for value, initial in zip(observed, starts):
        def residual(z):
            return offset + gain*z + (coupling + product*z)*z[right] + cubic*z**3 - value

        def jacobian(z):
            jac = np.diag(gain + product*z[right] + 3*cubic*z**2)
            jac[np.arange(width), right] += coupling + product*z
            return jac

        fit = root(residual, initial, jac=jacobian, tol=1e-10)
        z = fit.x
        if np.max(np.abs(residual(z))) > 1e-8:
            alternate = least_squares(residual, z, jac=jacobian).x
            if np.linalg.norm(residual(alternate)) < np.linalg.norm(residual(z)):
                z = alternate
        result.append(z)
    return np.asarray(result).reshape(observed.shape)


def _branches(panel, z):
    u = np.asarray(panel['u'])

    def pairs(x):
        return _intercept(x * np.roll(x, -1, axis=1))

    first = pairs(z) @ _ls(pairs(u), panel['v1'])
    target = np.asarray(panel['v2'])

    def cosine(x, p):
        return _intercept(np.column_stack((x, np.cos(p[0] * x + p[1:]))))

    def residual(p):
        design = cosine(u, p)
        return (design @ _ls(design, target) - target).ravel()

    parameters = least_squares(residual, np.r_[1.27, np.zeros(u.shape[1])],
                               max_nfev=150).x
    second = cosine(z, parameters) @ _ls(cosine(u, parameters), target)
    return first, second


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['x'])
    measured = np.vstack((train['x'], study['test']['x']))
    calibrated = _calibrate(study['panel_a'], measured)
    first, second = _branches(study['panel_c'], calibrated)
    cross = second[:, :, None] * first[:, None, :]
    scalar = _intercept(np.column_stack((first, second, np.trace(cross, axis1=1, axis2=2))))
    dynamic = _intercept(np.column_stack((first, second, cross.reshape(len(measured), -1))))
    return {'y_hat': scalar[n:] @ _ls(scalar[:n], train['y']),
            'unprojected': dynamic[n:] @ _ls(dynamic[:n], train['traj'])}
