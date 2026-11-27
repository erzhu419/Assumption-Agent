import importlib.util, json, pickle
import numpy as np

path = '/holdout_task/beta_m96_w04_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_beta_m96_w04_batch_04.pkl', 'rb') as f:
    sample = pickle.load(f)[0]
s, gt = sample['study'], sample['gt']
solve = m.solve_beta_m96_w04_batch_04
out = solve(s)
for key, reference_key in [('weights','w_opt'), ('terminal_field','terminal_field_A')]:
    ref = np.asarray(gt[reference_key]).reshape(out[key].shape)
    error = np.linalg.norm(out[key]-ref) / np.linalg.norm(ref)
    print('public', key, 'relative L2', error)
    assert error < 2e-5
cert = out['kkt_certificate']
gap = cert['primal_objective'] - cert['dual_lower_bound']
print('public KKT gap', gap)
assert 0 <= gap < 2e-4
sweep = {k: np.asarray(v) for k, v in s['sensor_calibration_sweeps'].items()}
banks = sweep['bank_id']
y_clean = m._calibrate(s)
worst = {}
# Every affected bank is tested alone and alongside the other bank in its wing;
# the reference remains the corresponding clean in-batch measurement.
for affected in [(0,), (1,), (2,), (3,), (0,1), (2,3)]:
    for case in ('missing', 'one-reference', 'three-spikes', 'one-spike-only'):
        trial = {k: v.copy() for k, v in sweep.items()}
        keep = np.ones(len(banks), dtype=bool)
        for b in affected:
            ix = np.flatnonzero(banks == b)
            if case == 'missing':
                keep[ix] = False
            elif case in ('one-reference', 'one-spike-only'):
                keep[ix[1:]] = False
                if case == 'one-spike-only':
                    trial['raw_readout'][ix[0]] += 0.5
            else:
                trial['raw_readout'][ix[[0, len(ix)//2, -1]]] += [0.5, -0.4, 0.7]
        study = dict(s, sensor_calibration_sweeps={k: v[keep] for k,v in trial.items()})
        y_err = np.max(np.abs(m._calibrate(study) - y_clean))
        result = solve(study)
        weight_err = np.linalg.norm(result['weights']-out['weights']) / np.linalg.norm(out['weights'])
        field_err = np.linalg.norm(result['terminal_field']-out['terminal_field']) / np.linalg.norm(out['terminal_field'])
        gap = result['kkt_certificate']['primal_objective'] - result['kkt_certificate']['dual_lower_bound']
        worst[case] = np.maximum(worst.get(case, np.zeros(4)), [y_err, weight_err, field_err, gap])
for case, errors in worst.items():
    print(case, '| max calibrated observation, relative weights, relative field, KKT gap:', *errors)
assert all((errors < [5e-6, 2e-5, 2e-5, 2e-4]).all() for errors in worst.values())
