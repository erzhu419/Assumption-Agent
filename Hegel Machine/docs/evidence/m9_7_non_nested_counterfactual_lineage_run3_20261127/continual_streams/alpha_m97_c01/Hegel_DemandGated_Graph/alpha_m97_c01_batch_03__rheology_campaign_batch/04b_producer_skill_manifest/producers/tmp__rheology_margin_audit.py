import importlib.util
import itertools
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'alpha_m97_c01_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for item in pickle.loads((root / 'public_alpha_m97_c01_batch_03.pkl').read_bytes()):
    d = item['dataset']
    assert d['shear_banding_index'] > .50
    tr = d['train']
    n = len(tr['lab_id'])
    a, h, x = (v[:n] for v in m._prepare(d))
    dt, t = d['past_dt'], d['eval_times']
    yc = tr['strain_curves_obs'] - tr['strain_0'][:, None]
    yr = tr['instant_rate_obs']
    for lab in np.unique(tr['lab_id']):
        k = tr['lab_id'] != lab
        z, w = m._fit(dt, t, a[k], h[k], x[k], yc[k], yr[k])
        c, r = m._basis(z, dt, t, a[~k], h[~k], x[~k])
        ec, er = c @ w - yc[~k], r @ w - yr[~k]
        complete = np.isfinite(tr['coupon_features_raw'][~k]).all(axis=1)
        print('held-out lab', lab, 'complete-feature signed median strain by time', np.round(np.median(ec[complete], axis=0), 4), 'signed median rate', round(np.median(er[complete]), 4))
        for level in np.unique(tr['stratum'][~k]):
            j = tr['stratum'][~k] == level
            print('  stratum', level, 'rows', j.sum(), 'median absolute strain/rate', np.round([np.median(abs(ec[j])), np.median(abs(er[j]))], 5))
    cal = d['calibration_load_cells']
    for lab in range(4):
        k = cal['lab_id'] == lab
        a, b = cal['cmd_stress'][k], cal['true_stress'][k]
        prior = m._prior_calibration(lab)
        drift = np.max(abs(b - np.polyval(prior, a)))
        worst = (0., None)
        missed = 0
        for pair in itertools.combinations(range(len(a)), 2):
            x, y = a[list(pair)], b[list(pair)]
            for j, delta in itertools.product(range(2), (-5., -3.01, 3.01, 5.)):
                obs = y.copy()
                obs[j] += delta
                coef = m._calibrate(x, obs, lab)
                err = np.max(abs(coef - prior))
                missed += err > .001
                if err > worst[0]:
                    worst = (err, (pair, j, delta, coef.round(6).tolist()))
        print('lab', lab, 'clean archive prediction max residual', round(drift, 6), 'missed two-point spikes', missed, 'worst', worst)
