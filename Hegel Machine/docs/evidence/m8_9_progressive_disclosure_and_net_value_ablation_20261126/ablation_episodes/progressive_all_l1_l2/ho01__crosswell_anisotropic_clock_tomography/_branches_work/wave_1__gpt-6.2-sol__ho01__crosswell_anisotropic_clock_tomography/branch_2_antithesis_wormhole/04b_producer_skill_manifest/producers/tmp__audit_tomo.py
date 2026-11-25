import pickle
import sys
import time
import numpy as np
sys.path.insert(0, '/holdout_task')
from tomography_solver import _RayModel, invert_plume_and_clocks

with open('/holdout_task/public_surveys.pkl', 'rb') as f:
    entries = pickle.load(f)
for num, entry in enumerate(entries, 1):
    s, gt = entry['survey'], entry['gt']
    a = _RayModel(s)
    start = time.perf_counter()
    out = invert_plume_and_clocks(s)
    elapsed = time.perf_counter() - start
    c = np.asarray(out['rcv_clock_bias_s'])
    loc = np.linalg.norm(np.asarray(out['plume_xz_m']) - gt['plume_xz_m'])
    clk = np.sqrt(np.mean((c - gt['rcv_clock_bias_s'])**2))
    assert set(out) == {'plume_xz_m', 'rcv_clock_bias_s'}
    assert len(out['plume_xz_m']) == 2 and c.shape == (a.nr,)
    assert all(type(x) is float and np.isfinite(x) for x in out['plume_xz_m'] + out['rcv_clock_bias_s'])
    assert abs(c.sum()) < 1e-12 and np.max(abs(c)) <= .004 + 1e-14
    assert loc <= 1.8 and clk <= .00025 and elapsed < 15
    cols = [0, *range(2, a.design.shape[1])]
    residuals = np.empty((2, a.ns, a.nr))
    fold_err = np.empty((2, a.ns, 2))
    for i in range(a.ns):
        train = dict(s, z_src_m=np.delete(s['z_src_m'], i), T_obs_s=np.delete(s['T_obs_s'], i, axis=0))
        held = slice(i*a.nr, (i+1)*a.nr)
        for k in range(2):
            m = _RayModel(train)
            design = a.design
            if k:
                m.design = m.design[:, cols]
                design = design[:, cols]
            p = m.solve()
            residuals[k, i] = a.y[held] - (design[held]@p[3:] + p[2]*a.kernel(*p[:2])[held])
            fold_err[k, i] = (np.linalg.norm(p[:2]-gt['plume_xz_m']), np.sqrt(np.mean((m.clock@p[(4 if k else 5):] - 1000*gt['rcv_clock_bias_s'])**2))*1000)
    # Use the same held-out picks for both models; the 2.5 ms gate lies between
    # the sub-0.1 ms inlier residuals and the observed ~5.5 ms skips.
    core = np.abs(residuals[0]) < 2.5
    steep = (np.abs(a.v) > .6).reshape(core.shape)
    rms = lambda e: 1000*np.sqrt(np.mean(e**2))
    print(f'public {num}: center {loc:.4f} m, clocks {clk*1e6:.2f} us, runtime {elapsed:.4f} s, held-out skips {np.count_nonzero(~core)}')
    for label, e, err in zip(('anisotropic', 'isotropic ablation'), residuals, fold_err):
        by_rcv = [(rms(e[core[:,j],j]), 1000*abs(np.mean(e[core[:,j],j]))) for j in range(a.nr)]
        print(label, 'held-out RMS all/shallow/steep (us)', [round(rms(e[mask]),2) for mask in (core, core & ~steep, core & steep)], 'worst receiver RMS/absolute mean (us)', np.round(np.max(by_rcv,axis=0),2), 'worst fold location (m)/clock (us)', np.round(err.max(axis=0),4))
