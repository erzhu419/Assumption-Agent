import numpy as np
from scipy.optimize import least_squares


def _scalar_design(x, z, frequency, ratio):
    return np.column_stack((np.ones(len(x)), z, z[:, 0] * z[:, 2],
                            z[:, 1] * z[:, 3], z[:, 0] * x[:, 1] ** 2,
                            np.sin(frequency * (x[:, 1] - ratio * x[:, 3]))))


def run_node(inputs: dict) -> dict:
    c, o = inputs['node_00'], inputs['node_01']
    x, xt = c['x_tr_cal'], c['x_te_cal']
    z, zt, y = o['z_tr'], o['z_te'], o['y']

    def residual(par):
        design = _scalar_design(x, z, *par)
        return design @ np.linalg.lstsq(design, y, rcond=None)[0] - y

    fit = min((least_squares(residual, (f, 1.), bounds=([.4, .2], [5., 2.]))
               for f in (1.3, 2.4, 3.6)), key=lambda result: result.cost)
    design = _scalar_design(x, z, *fit.x)
    y_hat = _scalar_design(xt, zt, *fit.x) @ np.linalg.lstsq(design, y, rcond=None)[0]
    design, query = np.c_[np.ones(len(z)), z], np.c_[np.ones(len(zt)), zt]
    state_pred = query @ np.linalg.lstsq(design, o['traj'], rcond=None)[0]
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
