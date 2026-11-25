import importlib.util
import pickle
import time
import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.legendre import leggauss

path = '/holdout_task/tomography_solver.py'
spec = importlib.util.spec_from_file_location('tomography_solver', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
q, w = leggauss(48)
u, w = (q + 1)/2, w/2
with open('/holdout_task/public_surveys.pkl','rb') as f:
    rows = pickle.load(f)
worst = np.zeros(3)
for k, row in enumerate(rows):
    s, truth = row['survey'], row['gt']
    times = s['T_obs_s']
    i, j = np.indices(times.shape)
    y = 1000 * times
    width = s['W_m']
    z0 = s['z_src_m'][:, None]
    dz = s['z_rcv_m'][None, :] - z0
    length = np.hypot(width, dz)
    radius = s['R_plume_m']
    paired = [[], []]
    for fold in range(5):
        test = (i + 2*j) % 5 == fold
        fit_times = np.where(test, np.nan, times)
        fit = mod.invert_plume_and_clocks({**s, 'T_obs_s': fit_times})
        x, z = fit['plume_xz_m']
        clocks = 1000 * np.asarray(fit['rcv_clock_bias_s'])
        d2 = (width*u - x)**2 + (z0[...,None] + dz[...,None]*u - z)**2
        plume = length * np.sum(w*np.exp(-d2/(2*radius**2)),axis=-1)
        basis = np.stack((length, plume, -dz**2/length),axis=-1)
        target = y - clocks
        ii, kk = np.triu_indices(len(z0), 1)
        # Within one receiver, the difference of two held-out picks cancels its clock exactly.
        pair_valid = test[ii] & test[kk]
        for m in range(2):
            a = basis[..., :2+m]
            start = np.linalg.lstsq(a[~test], target[~test],rcond=None)[0]
            p = least_squares(lambda p: a[~test] @ p - target[~test], start, loss='cauchy', f_scale=.3).x
            residual = y - a @ p
            paired[m].extend(np.abs(residual[ii] - residual[kk])[pair_valid])
    print('survey', k, 'held-out within-receiver median paired residual [isotropic, anisotropic] ms:', [float(np.median(v)) for v in paired])
    assert np.median(paired[1]) < np.median(paired[0]), 'anisotropy does not improve clock-free predictions'
    for index in np.ndindex(times.shape):
        t = times.copy()
        t[index] = np.nan
        now = time.perf_counter()
        out = mod.invert_plume_and_clocks({**s, 'T_obs_s': t})
        elapsed = time.perf_counter() - now
        xy, clocks = np.asarray(out['plume_xz_m']), np.asarray(out['rcv_clock_bias_s'])
        assert abs(clocks.sum()) < 1e-12 and np.max(np.abs(clocks)) <= .004 + 1e-14
        metrics = [np.linalg.norm(xy - truth['plume_xz_m']), 1000*np.sqrt(np.mean((clocks-truth['rcv_clock_bias_s'])**2)), elapsed]
        worst = np.maximum(worst, metrics)
print('leave-one-pick-out worst center m, clock RMSE ms, runtime s:', worst.tolist())
assert np.all(worst <= [1.8, .25, 15])
