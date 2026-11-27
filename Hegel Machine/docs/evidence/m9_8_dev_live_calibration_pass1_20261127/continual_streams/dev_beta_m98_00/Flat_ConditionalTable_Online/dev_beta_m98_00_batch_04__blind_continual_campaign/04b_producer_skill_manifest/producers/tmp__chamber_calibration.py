import numpy as np
from scipy.optimize import least_squares


def _calibration(sweeps):
    ref = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    bank = sweeps['single_bank_sweeps']
    ids = np.asarray(bank['bank_id'])
    signal = np.asarray(bank['ref_signal'])
    readout = np.asarray(bank['raw_readout'])
    bank_ref = np.eye(4)[ids] * signal[:, None]

    def design(x, j):
        return np.column_stack((np.ones(len(x)), x, x[:, j] * x[:, (j+1)%4], x[:, j]**3))

    def fit(j, keep):
        own = ids == j
        x = np.vstack((ref[keep], bank_ref[own]))
        y = np.r_[raw[keep, j], readout[own]]
        a = design(x, j)
        # Exclude the gross-offset cluster before estimating the sweep noise.
        good = abs(y) < 8
        c = np.linalg.lstsq(a[good], y[good], rcond=None)[0]
        r = a @ c - y
        inlier = good & (abs(r) < max(.001, 8*np.median(abs(r[good]))))
        if inlier.sum() >= a.shape[1] and np.linalg.matrix_rank(a[inlier]) == a.shape[1]:
            c = np.linalg.lstsq(a[inlier], y[inlier], rcond=None)[0]
        return c

    c = np.stack([fit(j, np.ones(len(ref), dtype=bool)) for j in range(4)], axis=1)

    def forward(x):
        return x @ c[1:5] + c[0] + c[5] * x * np.roll(x, -1, axis=1) + c[6] * x**3

    def invert(values):
        r = np.asarray(values, dtype=float).reshape(-1, 4)
        x = np.linalg.solve(c[1:5].T, (r-c[0]).T).T
        for _ in range(25):
            err = forward(x) - r
            if np.max(abs(err)) < 1e-12:
                break
            jac = np.broadcast_to(c[1:5].T, (len(x), 4, 4)).copy()
            j = np.arange(4)
            jac[:, j, j] += c[5]*np.roll(x, -1, axis=1) + 3*c[6]*x*x
            jac[:, j, (j+1)%4] += c[5]*x
            step = np.linalg.solve(jac, err[..., None])[..., 0]
            trial = x - step
            for _ in range(10):
                bad = np.sum((forward(trial)-r)**2, axis=1) > np.sum(err**2, axis=1)
                if not np.any(bad):
                    break
                step[bad] *= .5
                trial = x-step
            x = trial
        return x.reshape(np.shape(values))

    return invert, forward, fit
