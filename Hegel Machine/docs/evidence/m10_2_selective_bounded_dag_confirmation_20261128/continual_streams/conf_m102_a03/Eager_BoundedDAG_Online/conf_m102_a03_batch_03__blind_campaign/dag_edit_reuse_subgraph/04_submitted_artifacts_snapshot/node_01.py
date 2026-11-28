"""Correct the frozen affine alignment using the nonlinear reference panel."""
import numpy as np


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _basis(x, frequency):
    phase = x * frequency
    return np.column_stack((_affine(x), np.sin(phase), np.cos(phase)))


def _alignment(u, v):
    # Profile out the linear panel coefficients at each frequency update.
    frequency = np.full(u.shape[1], 2.9)
    ratio = (np.sqrt(5.) - 1.) / 2.

    def loss(j, value):
        frequency[j] = value
        b = _basis(u, frequency)
        r = b @ _fit(b, v) - v
        return np.sum(r * r)

    for _ in range(12):
        previous = frequency.copy()
        for j in range(len(frequency)):
            start = frequency[j]
            best = loss(j, start)
            lo, hi = 1.6, 4.2
            a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
            fa, fb = loss(j, a), loss(j, b)
            for _ in range(22):
                if fa < fb:
                    hi, b, fb = b, a, fa
                    a = hi - ratio * (hi - lo)
                    fa = loss(j, a)
                else:
                    lo, a, fa = a, b, fb
                    b = lo + ratio * (hi - lo)
                    fb = loss(j, b)
            point, score = (a, fa) if fa < fb else (b, fb)
            frequency[j] = point if score < best else start
        if np.max(np.abs(frequency - previous)) < 1.e-5:
            break
    return frequency, _fit(_basis(u, frequency), v)


def _readout(z):
    interaction = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    return np.column_stack((_affine(z), interaction))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    encoded = inputs['node_00']
    u = np.asarray(study['panel_b']['u'], dtype=float)
    v = np.asarray(study['panel_b']['v'], dtype=float)
    affine = _fit(_affine(u), v)
    frequency, mapping = _alignment(u, v)

    def correct(rows):
        raw = np.asarray(rows, dtype=float) - affine[0]
        x = np.linalg.solve(affine[1:].T, raw.T).T
        return _basis(x, frequency) @ mapping

    z, zt = correct(encoded['z']), correct(encoded['zt'])
    train = study['train']
    yh = _readout(zt) @ _fit(_readout(z), np.asarray(train['y'], dtype=float))
    trajectory = np.asarray(train['traj'], dtype=float)
    center = trajectory.mean(axis=0)
    _, _, vh = np.linalg.svd(trajectory - center, full_matrices=False)
    modes = vh[:2].T
    sp = center + (_affine(zt) @ _fit(_affine(z), (trajectory - center) @ modes)) @ modes.T
    diag = np.array([yh.mean(), yh.std(), sp[:, -1].mean(), sp[:, -1].std()])
    return {'y_hat': yh, 'state_pred': sp, 'diagnostics_vec': diag}
