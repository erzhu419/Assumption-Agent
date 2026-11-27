import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
sweep = s['sensor_calibration_sweeps']
x, y = sweep['ref_rosette'], sweep['raw_rosette']

def terms(x, i, variant):
    base = m._rosette_terms(x, i)
    own, nxt, opp, prev = x[:, (i + np.arange(4)) % 4].T
    if variant == 'pairwise':
        return np.column_stack((base, opp, (own - nxt)**2, (own - prev)**2, (own + opp)**2, own*nxt*prev))
    if variant == 'rational':
        q = own - nxt
        return np.column_stack((base, own/(1 + 0.25*q*q), (own + prev)/(1 + 0.25*own**2)))
    return base

for variant in ('current', 'pairwise', 'rational'):
    all_errors = []
    for test in np.array_split(np.random.default_rng(7).permutation(len(x)), 5):
        train = np.setdiff1d(np.arange(len(x)), test)
        A = np.column_stack((np.ones(len(train)), x[train]))
        e = np.linalg.norm(y[train] - A @ np.linalg.lstsq(A, y[train], rcond=None)[0], axis=1)
        clean = train[e <= max(6*np.median(e), 1e-12)]
        pred = np.column_stack([terms(x[test], i, variant) @ np.linalg.lstsq(terms(x[clean], i, variant), y[clean, i], rcond=None)[0] for i in range(4)])
        all_errors.extend(np.linalg.norm(y[test] - pred, axis=1))
    print(variant, 'out-of-fold calibration error p50/p90:', np.quantile(all_errors, [.5,.9]))

D, _ = m._design(s)
Z = s['Z_nuisance']
n_shift = len(s['mu_shifts'])
n_sensor = len(s['sensor_ij'])
assert len(D) == n_shift*n_sensor
shift = np.repeat(np.arange(n_shift), n_sensor)
sensor = np.tile(np.arange(n_sensor), n_shift)
metrics = {}
for label, groups in [('shift', shift), ('sensor', sensor)]:
    worst, rank_min = 0., Z.shape[1]
    for pulse in item['campaign_diagnostics']:
        delta = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw'])) - D @ pulse['w_pulse']
        for group in np.unique(groups):
            test = groups == group
            gamma, _, rank, _ = np.linalg.lstsq(Z[~test], delta[~test], rcond=None)
            rank_min = min(rank_min, rank)
            worst = max(worst, np.sqrt(np.mean((delta[test] - Z[test]@gamma)**2)))
    metrics[label] = worst
    print('held-out', label, 'worst pulse RMSE', worst, 'minimum nuisance rank', rank_min, '/', Z.shape[1])
print('study dimensions:', n_shift, 'shifts,', n_sensor, 'sensors;', 'batch/relay/label keys:', [k for k in s if any(t in k.lower() for t in ('batch','relay','label'))])
