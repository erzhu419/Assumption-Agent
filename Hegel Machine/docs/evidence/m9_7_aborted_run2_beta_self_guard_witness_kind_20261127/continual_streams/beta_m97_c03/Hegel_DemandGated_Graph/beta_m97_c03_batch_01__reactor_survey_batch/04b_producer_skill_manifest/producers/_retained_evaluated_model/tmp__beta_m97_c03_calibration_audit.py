import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_m97_c03_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cases = pickle.load(open('/holdout_task/public_beta_m97_c03_batch_01.pkl', 'rb'))
for c in cases:
    s, g = c['study'], c['gt']
    sw = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
    z = s['Z_nuisance']; gram = z.T@z + s['nuisance_ridge']*np.eye(z.shape[1])
    def profile(v):
        return v - z@np.linalg.solve(gram,z.T@v)
    a, _ = m._forward(s)
    print('projected design max error:', np.max(np.abs(profile(a)-g['S_tilde'])))
    y = m._calibrate(s); bank = np.asarray(s['row_bank_id']); raw = np.asarray(s['y_obs_raw'])
    for b in range(4):
        rows = bank == b
        print('bank',b,'profiled residual max (calibrated, uncalibrated):',*[np.max(np.abs(profile(v)[rows]-g['r_y'][rows])) for v in (y,raw)])
    # The documented direct-fit guarantee starts at four clean rows. Check that it
    # holds without history for both signs and every location of a near-threshold spike.
    worst = (0.0, None); failures = []
    for b in range(4):
        idx = np.flatnonzero(sw['bank_id']==b)
        for n in (5, len(idx)):
            for selection in (idx[:n], idx[-n:]):
                keep = (sw['bank_id']!=b); keep[selection] = True
                local = {k:v[keep].copy() for k,v in sw.items()}
                targets = np.flatnonzero(local['bank_id']==b)
                for j in targets:
                    for pulse in (-3.01, 3.01):
                        trial = dict(local,raw_readout=local['raw_readout'].copy())
                        trial['raw_readout'][j] += pulse
                        dev = np.max(np.abs(m._calibrate(dict(s,sensor_calibration_sweeps=trial))-y))
                        if dev > worst[0]: worst = (dev,(b,n,int(j),pulse))
                        if dev > 1e-6: failures.append((b,n,int(j),pulse,float(dev)))
    print('worst directly recoverable EMI case:',worst,'failures:',len(failures),'first:',failures[:3])
    v = np.asarray(s['past_vortex_circulation_trace']); critical = s['critical_vortex_circulation']
    for mode in (0,1):
        f = m._forward(dict(s,recirculation_loop_mode=mode,past_vortex_circulation_trace=v))[0]
        f0 = m._forward(dict(s,recirculation_loop_mode=mode,past_vortex_circulation_trace=np.full_like(v,critical)))[0]
        print('recirculation mode',mode,'above-vs-at-threshold forward response:',np.max(np.abs(f-f0)))
        assert (np.max(np.abs(f-f0)) > 0) == (mode == 0)
