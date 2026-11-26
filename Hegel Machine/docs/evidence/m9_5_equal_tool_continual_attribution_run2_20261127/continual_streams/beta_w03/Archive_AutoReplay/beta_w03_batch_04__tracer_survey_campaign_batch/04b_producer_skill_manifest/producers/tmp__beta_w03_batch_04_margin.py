import pickle, runpy
import numpy as np

f = runpy.run_path('/holdout_task/beta_w03_batch_04_solver.py')
with open('/holdout_task/public_beta_w03_batch_04.pkl', 'rb') as fp:
    item = pickle.load(fp)[0]
s, gt = item['study'], item['gt']
bank, ref, raw = (np.asarray(s['sensor_calibration_sweeps'][k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
rows = np.asarray(s['row_bank_id'])
S, y = gt['S_tilde'], gt['r_y']
base = f['_calibrate'](s)
records = []
for b in range(4):
    wing = b // 2
    ids = np.flatnonzero(bank == b)
    mask = (bank // 2 != wing)
    for anchor in ids:
        keep = mask.copy()
        keep[anchor] = True
        for delta in (-0.18, 0., 0.18):
            altered = raw.copy()
            altered[anchor] += delta
            case = dict(s, sensor_calibration_sweeps=dict(bank_id=bank[keep], ref_signal=ref[keep], raw_readout=altered[keep]))
            error = np.max(np.abs(f['_calibrate'](case)[rows // 2 == wing] - base[rows // 2 == wing]))
            records.append((error, b, int(anchor), delta, case))
for regime, selected in [('clean spot', [r for r in records if r[3] == 0]), ('EMI spot', [r for r in records if r[3] != 0])]:
    error, b, anchor, delta, case = max(selected, key=lambda r: r[0])
    out = f['solve_beta_w03_batch_04'](case)
    w, c = out['weights'], out['kkt_certificate']
    r = S @ w - y
    grad = S.T @ r + s['source_ridge'] * w
    p = .5 * (r @ r + s['source_ridge'] * (w @ w))
    nu = c['nu_sum']
    lb = p - grad @ w + nu * s['total_budget'] + s['u_box'] * np.minimum(grad - nu, 0).sum()
    print(regime, 'slices', len(selected), 'bank/anchor/EMI', (b, anchor, delta), 'max corrected observation drift', error, 'weight relative error', np.linalg.norm(w-gt['w_opt'])/np.linalg.norm(gt['w_opt']), 'field relative error', np.linalg.norm(out['terminal_field']-gt['terminal_field_A'])/np.linalg.norm(gt['terminal_field_A']), 'external gap', p-min(lb,c['dual_lower_bound']))
