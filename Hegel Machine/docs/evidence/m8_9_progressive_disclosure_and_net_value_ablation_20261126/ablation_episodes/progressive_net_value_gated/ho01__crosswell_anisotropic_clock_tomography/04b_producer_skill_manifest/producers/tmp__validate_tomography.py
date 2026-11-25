import importlib.util
import pickle
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from pathlib import Path

path = Path('/holdout_task/tomography_solver.py')
spec = importlib.util.spec_from_file_location('tomography_solver', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rows = pickle.loads(path.with_name('public_surveys.pkl').read_bytes())
q, w = leggauss(64)
u, w = (q + 1) / 2, w / 2

def terms(s, center):
    z0 = np.asarray(s['z_src_m'])[:, None]
    dz = np.asarray(s['z_rcv_m'])[None, :] - z0
    width = s['W_m']
    length = np.hypot(width, dz)
    radius = s['R_plume_m']
    d2 = (width * u - center[0])**2 + (z0[..., None] + dz[..., None] * u - center[1])**2
    return np.stack((length, -dz**2 / length, length * np.sum(w * np.exp(-d2 / (2 * radius**2)), axis=-1)), axis=-1)

maximum = {}
def check(group, s, center, clocks):
    now = time.perf_counter()
    out = mod.invert_plume_and_clocks(s)
    sec = time.perf_counter() - now
    assert set(out) == {'plume_xz_m', 'rcv_clock_bias_s'}
    assert all(isinstance(x, list) and all(type(v) is float for v in x) for x in out.values())
    c = np.asarray(out['rcv_clock_bias_s'])
    xy = np.asarray(out['plume_xz_m'])
    assert c.shape == np.shape(clocks) and xy.shape == (2,)
    assert abs(c.sum()) < 1e-12 and np.max(np.abs(c)) <= 0.004 + 1e-14
    assert np.all((xy >= 0) & (xy <= [s['W_m'], s['H_m']]))
    metrics = np.array([np.linalg.norm(xy - center), 1000 * np.sqrt(np.mean((c - clocks)**2)), sec])
    maximum[group] = np.maximum(maximum.get(group, 0), metrics)

for row in rows:
    s, gt = row['survey'], row['gt']
    center, clocks = gt['plume_xz_m'], gt['rcv_clock_bias_s']
    check('public', s, center, clocks)
    y = (s['T_obs_s'] - clocks).ravel()
    a = terms(s, center).reshape(-1, 3)
    p = least_squares(lambda p: 1000 * (a @ p - y), np.linalg.lstsq(a, y, rcond=None)[0], loss='cauchy', f_scale=.2).x
    err = 1000 * (y - a @ p)
    sig = 1.4826 * np.median(np.abs(err - np.median(err)))
    print('public background s0, epsilon, plume delta_s, noise MAD ms, >3ms residuals:', p[0], p[1]/p[0], p[2], sig, (abs(err)>3).sum())
    for i in range(len(s['z_src_m'])):
        check('public source jackknife', {**s, 'z_src_m': np.delete(s['z_src_m'], i), 'T_obs_s': np.delete(s['T_obs_s'], i, axis=0)}, center, clocks)
    for j in range(len(clocks)):
        keep = np.arange(len(clocks)) != j
        # The reduced array has a new clock gauge; emulate that acquisition
        # by subtracting the mean of the retained true offsets from all picks.
        mean = clocks[keep].mean()
        check('public receiver jackknife', {**s, 'z_rcv_m': s['z_rcv_m'][keep], 'T_obs_s': s['T_obs_s'][:, keep] - mean}, center, clocks[keep] - mean)
    rng = np.random.default_rng(283)
    for k in range(36):
        W = [32., 40., 52.][k % 3]
        H = [70., 80., 100.][k // 3 % 3]
        n, m = [(11, 12), (15, 16), (19, 21)][k // 9 % 3]
        ss = {**s, 'W_m': W, 'H_m': H, 'z_src_m': np.linspace(.075*H, .925*H, n), 'z_rcv_m': np.linspace(.063*H, .937*H, m)}
        xz = np.array([rng.uniform(.13, .87)*W, rng.uniform(.14, .86)*H])
        b = rng.uniform(-1, 1, m); b -= b.mean(); b *= .0039 / np.max(np.abs(b))
        pp = p * [rng.uniform(.9, 1.1), rng.uniform(.55, 1.35), rng.uniform(.65, 1.35)]
        y = terms(ss, xz) @ pp + b + rng.normal(0, sig * .001 * (2 if k % 4 == 0 else 1), (n, m))
        skip = rng.choice(y.size, round(.04*y.size), replace=False)
        y.flat[skip] += .006 * rng.choice([-1, 1], len(skip))
        ss['T_obs_s'] = y
        check('independent-quadrature stress', ss, xz, b)
for group, metric in maximum.items():
    print(group, '(worst center m, clock RMSE ms, runtime s):', metric.tolist())
assert all(np.all(metric <= [1.8, .25, 15.]) for metric in maximum.values()), 'At least one stress slice exceeded a target'
