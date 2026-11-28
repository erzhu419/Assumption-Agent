import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _warp(x, params):
    frequency, amplitude = params
    return x + amplitude * np.sin(frequency * x)


def _profile(x, target, start):
    def residual(params):
        design = _affine(_warp(x, params))
        weights = np.linalg.lstsq(design, target, rcond=None)[0]
        return (design @ weights - target).ravel()

    return least_squares(residual, start, bounds=([1.5, -.8], [4.5, 0.])).x


def _scalar_design(x, z, modulation):
    return np.column_stack((_affine(z), z[:, 0] * z[:, 2], z[:, 1] * z[:, 3],
                            z[:, 0] * modulation ** 2))


def run_node(inputs: dict) -> dict:
    c, o = inputs['node_00'], inputs['node_01']
    x, xt = c['x_tr_cal'], c['x_te_cal']
    z, zt, y, traj = o['z_tr'], o['z_te'], o['y'], o['traj']

    # Recover the incoming operator's affine mixing without accessing its calibration panel.
    incoming = _profile(x, z, (3., -.4))
    mixing = np.linalg.lstsq(_affine(_warp(x, incoming)), z, rcond=None)[0][1:]
    # Trajectory labels identify the shared coordinate correction on training rows.
    corrected = _profile(x, traj, incoming)
    u, ut = _warp(x, corrected), _warp(xt, corrected)
    z = z + (u - _warp(x, incoming)) @ mixing
    zt = zt + (ut - _warp(xt, incoming)) @ mixing

    base = _scalar_design(x, z, u[:, 1])
    query = _scalar_design(xt, zt, ut[:, 1])

    def contrast(coords, params):
        return np.sin(params[0] * coords[:, 1] - params[1] * coords[:, 3])

    def residual(params):
        design = np.column_stack((base, contrast(x, params)))
        return design @ np.linalg.lstsq(design, y, rcond=None)[0] - y

    fit = min((least_squares(residual, (f, f), bounds=([.4, .3], [5., 5.]))
               for f in (1.2, 2.2, 3.6)), key=lambda result: result.cost)
    design = np.column_stack((base, contrast(x, fit.x)))
    query = np.column_stack((query, contrast(xt, fit.x)))
    y_hat = query @ np.linalg.lstsq(design, y, rcond=None)[0]
    state_pred = _affine(ut) @ np.linalg.lstsq(_affine(u), traj, rcond=None)[0]
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
