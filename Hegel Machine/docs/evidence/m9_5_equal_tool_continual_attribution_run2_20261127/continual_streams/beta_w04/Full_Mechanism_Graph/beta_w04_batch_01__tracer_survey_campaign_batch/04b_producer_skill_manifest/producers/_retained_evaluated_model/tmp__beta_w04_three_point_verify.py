import importlib.util, itertools, pickle
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root/'beta_w04_batch_01_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ns = {}; exec((root/'validate_public_schema.py').read_text().split('\n_d =', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']
item = pickle.load(open(root/'public_beta_w04_batch_01.pkl', 'rb'))[0]
s, gt = item['study'], item['gt']
bank, ref, raw = m._sweep_arrays(s['sensor_calibration_sweeps'])
rows, obs = s['row_bank_id'], s['y_obs_raw']
worst_gap = worst_weight = worst_field = worst_cal = 0.
invalid = 0
for gain, offset in [(1.,0.),(1.18,.07),(.83,-.07),(1.18,-.07),(.83,.07)]:
    for spike in (0., .8, -.8):
        for b in range(4):
            ids = np.flatnonzero(bank == b); other = np.flatnonzero(bank != b); mask = rows == b
            y = obs.copy(); y[mask] = gain*obs[mask] + offset
            worst = (-1., None)
            for triple in itertools.combinations(ids, 3):
                ix = np.r_[other, triple]
                z = np.r_[raw[other], gain*raw[list(triple)]+offset]
                for j in range(3) if spike else (0,):
                    readings = z.copy(); readings[len(other)+j] += spike
                    t = dict(s, y_obs_raw=y, sensor_calibration_sweeps=dict(bank_id=bank[ix], ref_signal=ref[ix], raw_readout=readings))
                    err = np.max(np.abs(m._calibrate(t)[mask] - obs[mask]))
                    if err > worst[0]: worst = (err, t)
            err, t = worst
            out = m.solve_beta_w04_batch_01(t)
            certs = [verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], s['u_box'], budget, c, gap_tol=1e-6) for budget,c in zip(s['budget_grid'],out['kkt_certificates'])]
            if not all(c['valid'] for c in certs):
                invalid += 1; print('failure', gain, offset, spike, b, err, certs)
            worst_cal = max(worst_cal,err)
            worst_gap = max(worst_gap,*(c['gap'] for c in certs))
            worst_weight = max(worst_weight,np.max(np.abs(out['weights_by_budget']-gt['weights_by_budget'])))
            worst_field = max(worst_field,np.linalg.norm(out['terminal_field_mid_budget']-gt['terminal_field_mid_budget'])/np.linalg.norm(gt['terminal_field_mid_budget']))
print('worst calibration, certificate gap, weight, relative field; failed slices:',worst_cal,worst_gap,worst_weight,worst_field,invalid)
assert invalid == 0 and worst_cal < 2e-5
