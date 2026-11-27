import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/beta_m96_w01_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_beta_m96_w01_batch_04.pkl', 'rb') as f:
    case = pickle.load(f)[0]
s, gt = case['study'], case['gt']
q = s['sensor_calibration_sweeps']
ids = np.asarray(q['bank_id'])
S, y = gt['S_tilde'], gt['r_y']
ridge, cap, budget = (s[k] for k in ('source_ridge', 'u_box', 'total_budget'))
for bank in np.unique(ids):
    select = np.flatnonzero(ids == bank)
    other = np.flatnonzero(ids != bank)
    worst = np.zeros(4)
    count = 0
    for pair in itertools.combinations(select, 2):
        for spike in (None, pair[0], pair[1]):
            keep = np.r_[other, pair]
            sweeps = {k: np.asarray(v)[keep].copy() for k, v in q.items()}
            if spike is not None:
                sweeps['raw_readout'][len(other) + pair.index(spike)] += 0.22
            out = m.solve_beta_m96_w01_batch_04(dict(s, sensor_calibration_sweeps=sweeps))
            w, cert = out['weights'], out['kkt_certificate']
            r = S @ w - y
            j = 0.5 * (r @ r + ridge * (w @ w))
            g = S.T @ r + ridge * w
            nu = cert['nu_sum']
            low = j - g @ w + nu * budget + cap * np.minimum(g - nu, 0).sum()
            stationarity = np.max(np.abs(g - nu - cert['lam_lower'] + cert['lam_upper']))
            field_err = np.linalg.norm(out['terminal_field'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
            worst = np.maximum(worst, [field_err, j - min(low, cert['dual_lower_bound']), stationarity, abs(j - cert['primal_objective'])])
            count += 1
    print(f'bank {bank}: {count} sparse slices; max field, gap, stationarity, objective error: {worst}', flush=True)
    assert np.all(worst < [1e-5, 1e-6, 1e-4, 1e-6])
