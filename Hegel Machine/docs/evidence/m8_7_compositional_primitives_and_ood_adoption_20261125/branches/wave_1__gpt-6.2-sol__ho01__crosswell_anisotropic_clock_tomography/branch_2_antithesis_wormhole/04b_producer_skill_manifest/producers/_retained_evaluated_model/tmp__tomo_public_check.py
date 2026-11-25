import pickle
import time
import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.legendre import leggauss
from tomography_solver import invert_plume_and_clocks


def ray_terms(s, center):
    # Independent Gauss-Legendre quadrature, not the solver's analytic kernel.
    zs = np.asarray(s['z_src_m'])[:, None]
    dz = np.asarray(s['z_rcv_m'])[None, :] - zs
    w = s['W_m']
    length = np.hypot(w, dz)
    nodes, weights = leggauss(32)
    frac = (nodes + 1) / 2
    d2 = (w * frac - center[0])**2 + (zs[..., None] + dz[..., None] * frac - center[1])**2
    g = length * (np.exp(-0.5 * d2 / s['R_plume_m']**2) @ (weights / 2))
    return np.stack((length, length * (dz / length)**2, g), axis=-1)


def public_cases():
    with open('/holdout_task/public_surveys.pkl','rb') as f:
        return pickle.load(f)


def run():
    for i, d in enumerate(public_cases()):
        s, gt = d['survey'], d['gt']
        y = np.asarray(s['T_obs_s']) * 1e3
        y0 = y - 1e3 * np.asarray(gt['rcv_clock_bias_s'])
        truth_terms = ray_terms(s, gt['plume_xz_m'])
        q = least_squares(lambda p: (truth_terms @ p - y0).ravel(), [.4,-.035,.15], loss='cauchy', f_scale=.15).x
        r = truth_terms @ q - y0
        print('public', i, 's0, eps, delta_s in ms/m', q[0], -q[1]/q[0], q[2], 'inlier MAD ms', 1.4826*np.median(abs(r-np.median(r))), 'outlier count', np.sum(abs(r)>3), flush=True)
        results = []
        for j in range(y.shape[0]):
            subs = dict(s)
            obs = np.array(s['T_obs_s'], dtype=float, copy=True)
            obs[j] = np.nan
            subs['T_obs_s'] = obs
            start = time.perf_counter()
            out = invert_plume_and_clocks(subs)
            elapsed = time.perf_counter() - start
            center = np.asarray(out['plume_xz_m'])
            clocks = np.asarray(out['rcv_clock_bias_s'])
            assert set(out)=={'plume_xz_m', 'rcv_clock_bias_s'} and len(clocks)==y.shape[1]
            assert abs(clocks.sum())<1e-12 and max(abs(clocks))<=.004+1e-12
            terms = ray_terms(s,center)
            train = np.arange(len(y)) != j
            q_fit = least_squares(lambda p: (terms[train] @ p + clocks*1e3 - y[train]).ravel(), [.4,-.035,.15], loss='cauchy', f_scale=.15).x
            pred = terms[j] @ q_fit + clocks*1e3
            ordinary = abs(r[j]) < 3
            results.append([np.linalg.norm(center-gt['plume_xz_m']), np.sqrt(np.mean((clocks-gt['rcv_clock_bias_s'])**2)), np.sqrt(np.mean((pred[ordinary]-y[j,ordinary])**2)), elapsed])
        a = np.asarray(results)
        print('leave-one-source-out maxima: center m, clock RMSE s, ordinary held-ray RMSE ms, runtime s:', a.max(axis=0), 'source indices:', np.argmax(a, axis=0), flush=True)
        assert np.all(a.max(axis=0)[[0,1,3]] < [1.8,.00025,15])


if __name__ == '__main__':
    run()
