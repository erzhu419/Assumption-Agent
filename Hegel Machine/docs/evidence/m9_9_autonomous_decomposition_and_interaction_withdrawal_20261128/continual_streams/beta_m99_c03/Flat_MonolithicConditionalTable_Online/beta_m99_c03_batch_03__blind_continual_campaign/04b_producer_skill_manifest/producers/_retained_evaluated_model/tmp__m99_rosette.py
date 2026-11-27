import numpy as np


def _rosette_design(ref, j):
    own, nxt, prev = ref[:, j], ref[:, (j + 1) % 4], ref[:, (j - 1) % 4]
    return np.column_stack((np.ones(len(ref)), own, nxt, prev, own*nxt, own**3))


def _fit_rosette(sweeps):
    ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    banks = sweeps['single_bank_sweeps']
    coef = []
    for j in range(4):
        take = banks['bank_id'] == j
        axes = np.zeros((np.count_nonzero(take), 4))
        axes[:, j] = banks['ref_signal'][take]
        design = _rosette_design(np.vstack((ref, axes)), j)
        target = np.r_[raw[:, j], banks['raw_readout'][take]]
        clean = np.abs(target) < 8.0
        coef.append(np.linalg.lstsq(design[clean], target[clean], rcond=None)[0])
    return np.asarray(coef)


def _calibrated_observations(study):
    coef = _fit_rosette(study['sensor_calibration_sweeps'])
    meas = study['y_obs_raw'].reshape(-1, 4)
    j = np.arange(4)
    nxt, prev = (j + 1) % 4, (j - 1) % 4
    linear = np.zeros((4, 4))
    linear[j, j], linear[j, nxt], linear[j, prev] = coef[:, 1:4].T
    x = np.linalg.solve(linear, (meas - coef[:, 0]).T).T
    for _ in range(20):
        residual = x @ linear.T + coef[:, 0] + coef[:, 4]*x*x[:, nxt] + coef[:, 5]*x**3 - meas
        if np.max(np.abs(residual)) < 1e-13:
            return x.ravel()
        jac = np.broadcast_to(linear, (len(x), 4, 4)).copy()
        jac[:, j, j] += coef[:, 4]*x[:, nxt] + 3*coef[:, 5]*x**2
        jac[:, j, nxt] += coef[:, 4]*x
        x -= np.linalg.solve(jac, residual[..., None])[..., 0]
    raise RuntimeError('Sensor rosette inversion did not converge')
