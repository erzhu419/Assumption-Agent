import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _label_basis(x, z, frequency):
    a, b, c, d = z.T
    return np.column_stack((_affine(z), a * c, b * d,
                            np.sin(frequency * (x[:, 2] - d))))


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    cal = inputs['node_00']
    mix = inputs['node_01']
    x, xt = cal['cal_tr'], cal['cal_te']
    z, zt = mix['z_tr'], mix['z_te']
    y = train['y']

    # Profile out the linear coefficients when estimating the sine frequency.
    frequency, best_loss = 2.5, np.inf
    for step in (0.10, 0.01):
        center = frequency
        for k in center + step * np.arange(-10, 11):
            basis = _label_basis(x, z, k)
            coef = _fit(basis, y)
            residual = basis @ coef - y
            loss = residual @ residual
            if loss < best_loss:
                frequency, best_loss, label_coef = k, loss, coef
    y_hat = _label_basis(xt, zt, frequency) @ label_coef

    state_coef = _fit(_affine(x), train['traj'])
    state_pred = _affine(xt) @ state_coef
    terminal = state_pred[:, -1]
    diagnostics = np.array((y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()))
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
