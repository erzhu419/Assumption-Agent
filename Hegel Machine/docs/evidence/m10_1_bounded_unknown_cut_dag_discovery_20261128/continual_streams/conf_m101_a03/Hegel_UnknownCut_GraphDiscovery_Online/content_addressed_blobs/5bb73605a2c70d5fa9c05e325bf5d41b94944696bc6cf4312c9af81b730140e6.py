import numpy as np


def _basis(x, frequency, amplitude):
    return np.column_stack((np.ones(len(x)), x + amplitude * np.sin(frequency * x)))


def _fit_panel(u, v):
    # Profile out the affine mixing for each shared coordinate warp.
    def evaluate(frequency, amplitude):
        design = _basis(u, frequency, amplitude)
        weights = np.linalg.lstsq(design, v, rcond=None)[0]
        error = np.sum((design @ weights - v) ** 2)
        return error, frequency, amplitude, weights

    best = min((evaluate(f, a) for f in np.linspace(1.5, 4.5, 13)
                for a in np.linspace(-0.8, 0.0, 9)), key=lambda item: item[0])
    df, da = 0.125, 0.05
    for _ in range(48):
        _, f, a, _ = best
        trial = min((evaluate(f + i * df, a + j * da)
                     for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j),
                    key=lambda item: item[0])
        if trial[0] < best[0]:
            best = trial
        else:
            df *= 0.5
            da *= 0.5
        if df < 1e-6:
            break
    return best[1:]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    calibrated = inputs['node_00']
    panel = study['panel_b']
    frequency, amplitude, weights = _fit_panel(panel['u'], panel['v'])
    return {
        'z_tr': _basis(calibrated['x_tr_cal'], frequency, amplitude) @ weights,
        'z_te': _basis(calibrated['x_te_cal'], frequency, amplitude) @ weights,
        'y': study['train']['y'],
        'traj': study['train']['traj'],
    }
