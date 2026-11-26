import itertools, pickle, runpy
from pathlib import Path
import numpy as np
m = runpy.run_path('beta_w03_batch_03_solver.py')
verify = {}
exec(Path('validate_public_schema.py').read_text().split('cand_p =')[0].split('_d =')[0], verify)
check = verify['verify_box_simplex_kkt_certificate']
q = pickle.load(open('public_beta_w03_batch_03.pkl', 'rb'))[0]
s, gt = q['study'], q['gt']
sw = s['sensor_calibration_sweeps']
ids, ref = sw['bank_id'], sw['ref_signal']
row = s['row_bank_id']
profiles = np.array(m['_ARCHIVE_PROFILES'])
nonarch = np.array([[1.12,.079],[.92,-.073],[.86,.046],[1.19,-.12]])
settings = {'identity':profiles[:,0], 'archive':profiles[:,-1], 'new':nonarch}
for name, coeff in settings.items():
    gain, offset = coeff.T
    raw = gain[ids]*sw['raw_readout']+offset[ids]
    obs = gain[row]*s['y_obs_raw']+offset[row]
    scenarios=[('full', np.ones(len(ids),bool), ()), ('emi', np.ones(len(ids),bool), tuple(np.flatnonzero(ids==2)[[0,4,9]]))]
    for withheld in [(0,), (1,), (2,), (3,), (0,1), (2,3)]:
        for n in (0,1):
            keep = np.ones(len(ids),bool)
            for b in withheld:
                idx = np.flatnonzero(ids==b)
                keep[idx[n:]] = False
            scenarios.append((f'{withheld}/{n}',keep,()))
    worst = {}
    for tag, keep, emi in scenarios:
        values = raw.copy()
        if emi: values[list(emi)] += [0.60,-0.48,0.83]
        st = dict(s, y_obs_raw=obs, sensor_calibration_sweeps={'bank_id':ids[keep], 'ref_signal':ref[keep], 'raw_readout':values[keep]})
        out=m['solve_beta_w03_batch_03'](st)
        checks=[check(gt['S_tilde'],gt['r_y'],s['source_ridge'],s['u_box'],float(b),c,gap_tol=2e-4) for b,c in zip(s['budget_grid'],out['kkt_certificates'])]
        err = float(np.max(np.abs(out['weights_by_budget']-gt['weights_by_budget'])))
        field = float(np.linalg.norm(out['terminal_field_mid_budget']-gt['terminal_field_mid_budget'])/np.linalg.norm(gt['terminal_field_mid_budget']))
        gap = max(c['gap'] for c in checks)
        group = 'full_or_emi' if tag in ('full','emi') else 'sparse'
        if group not in worst or err>worst[group][0]: worst[group]=(err,field,gap,tag)
        if not all(c['valid'] for c in checks): print('certificate failure', name, tag, checks)
    print(name,'worst by max weight error (weight,field,gap,slice)',worst)
