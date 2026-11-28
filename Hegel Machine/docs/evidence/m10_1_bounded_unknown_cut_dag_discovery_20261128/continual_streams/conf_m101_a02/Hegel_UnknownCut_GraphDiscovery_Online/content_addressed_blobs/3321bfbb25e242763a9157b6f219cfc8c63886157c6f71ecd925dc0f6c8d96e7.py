import numpy as np


def _features(x, z):
    cyclic = x * np.roll(z, 1, axis=1)
    base = np.column_stack((np.ones(len(x)), z, cyclic))
    contrast = z[:, 2:].sum(axis=1) - z[:, :2].sum(axis=1)
    # Two compact saturation models: one gated by the cyclic interaction,
    # the other by the positive half of the mixed-coordinate contrast.
    interaction_gate = 2 * cyclic[:, 0] - cyclic.sum(axis=1)
    mixed_gate = z[:, 2:].sum(axis=1)
    gates = np.column_stack((interaction_gate, mixed_gate))
    return base, contrast[:, None] / (1 + gates**2)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    cal, mix = inputs['node_00'], inputs['node_01']
    f, corrections = _features(cal['cal_tr'], mix['z_tr'])
    g, test_corrections = _features(cal['cal_te'], mix['z_te'])
    y_train = study['train']['y']
    predictions = []
    for a, b in zip(corrections.T, test_corrections.T):
        fit = np.column_stack((f, a))
        coefficients = np.linalg.lstsq(fit, y_train, rcond=None)[0]
        predictions.append(g @ coefficients[:-1] + b * coefficients[-1])
    y = np.mean(predictions, axis=0)
    state = g @ np.linalg.lstsq(f, study['train']['traj'], rcond=None)[0]
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
