import itertools
import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    s = item['study']
    sweeps = s['sensor_calibration_sweeps']
    x, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
    design = np.c_[np.ones(len(x)), x]
    splits = np.array_split(np.arange(len(x)), 5)
    pairs = [(i, j, sign) for i, j in itertools.combinations(range(4), 2) for sign in (-1, 1)]
    candidates = [design]
    for i, j, sign in pairs:
        q = x[:, i] + sign*x[:, j]
        candidates.append(np.c_[design, q*q])
        for a in (0.01, 0.1, 1.0):
            candidates.append(np.c_[design, q/(1+a*q*q)])
    def predict(D, train, test):
        return D[test] @ np.linalg.lstsq(D[train], raw[train], rcond=None)[0]
    worst = np.zeros(2)
    for k, test in enumerate(splits):
        inner = [f for j, f in enumerate(splits) if j != k]
        train = np.concatenate(inner)
        def score(D):
            return sum(np.sum((predict(D, np.concatenate([f for j, f in enumerate(inner) if j != l]), val)-raw[val])**2) for l, val in enumerate(inner))
        chosen = min(candidates, key=score)
        for j, D in enumerate((design, chosen)):
            worst[j] = max(worst[j], np.sqrt(np.mean((predict(D, train, test)-raw[test])**2, axis=0)).max())
    print('Worst outer-fold channel RMSE, affine vs nested pair/rational selection:', worst.tolist())
    single = sweeps['single_bank_sweeps']
    gain, bias = np.empty(4), np.empty(4)
    for j in range(4):
        mask = single['bank_id'] == j
        gain[j], bias[j] = np.linalg.lstsq(np.c_[single['ref_signal'][mask], np.ones(mask.sum())], single['raw_readout'][mask], rcond=None)[0]
    S, _ = m['_forward'](s)
    worst = np.zeros(2)
    wall = ((s['sensor_ij'] == 0) | (s['sensor_ij'] == np.array([s['nx']-1,s['ny']-1]))).any(axis=1)
    for p in item.get('campaign_diagnostics') or []:
        obs = p['y_pulse_raw']
        pred = S@p['w_pulse']
        calibrations = (m['_calibrate'](dict(s,y_obs_raw=obs)), ((obs.reshape(-1,4)-bias)/gain).ravel())
        for i, y in enumerate(calibrations):
            e = (pred-y).reshape(len(s['mu_shifts']),-1)
            worst[i] = max(worst[i], max(np.sqrt(np.mean(e[:,mask]**2,axis=1)).max() for mask in (wall,~wall) if mask.any()))
    print('Worst measured pulse shift/wall slice RMSE, full rosette vs single-bank fit:', worst.tolist())
