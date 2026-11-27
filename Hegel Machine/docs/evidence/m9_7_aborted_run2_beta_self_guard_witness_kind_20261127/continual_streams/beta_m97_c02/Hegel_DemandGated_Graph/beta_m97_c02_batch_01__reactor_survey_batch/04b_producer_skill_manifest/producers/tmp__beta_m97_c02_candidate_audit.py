import importlib.util
import itertools
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', '/tmp/beta_m97_c02_candidate.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for item in pickle.load(open(root+'public_beta_m97_c02_batch_01.pkl','rb')):
    s, gt = item['study'], item['gt']
    a = {k: np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}
    ids = a['bank_id']
    rows = np.asarray(s['row_bank_id'])
    raw = np.asarray(s['y_obs_raw'])
    z = s['Z_nuisance']
    projector = np.eye(len(raw)) - z @ np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]), z.T)
    base = m._calibration_parameters(a)
    for n in (2,3,4,5):
        worst = {'clean': (0., None), 'EMI': (0., None)}
        for bank in range(4):
            inds = np.flatnonzero(ids==bank)
            others = np.flatnonzero(ids!=bank)
            # Retain the full, widest-spaced and consecutive observed spans.
            ordered = inds[np.argsort(a['ref_signal'][inds])]
            subsets = {tuple(ordered[np.round(g).astype(int)]) for g in [np.linspace(0,len(inds)-1,n)]}
            subsets.update(tuple(ordered[j:j+n]) for j in range(len(inds)-n+1))
            for subset in sorted(subsets):
                keep = np.r_[others, subset]
                sweep = {k:v[keep] for k,v in a.items()}
                shifts = [(-1,0.)]
                if n >= 3:
                    shifts += list(itertools.product(range(n),(-4.,4.)))
                for j, shift in shifts:
                    trial = {**sweep, 'raw_readout':sweep['raw_readout'].copy()}
                    if shift: trial['raw_readout'][len(others)+j] += shift
                    params = m._calibration_parameters(trial)
                    calibrated = (raw-params[rows,1])/params[rows,0]
                    err = float(np.max(np.abs(projector@calibrated-gt['r_y'])))
                    kind = 'EMI' if shift else 'clean'
                    if err>worst[kind][0]:
                        worst[kind]=(err,(bank,subset,j,shift,params[bank].tolist(),base[bank].tolist()))
        for kind,(err,witness) in worst.items():
            if witness is not None:
                print(f'{n} sweeps, {kind}: worst projected target max error {err:.3e}; witness {witness}')
