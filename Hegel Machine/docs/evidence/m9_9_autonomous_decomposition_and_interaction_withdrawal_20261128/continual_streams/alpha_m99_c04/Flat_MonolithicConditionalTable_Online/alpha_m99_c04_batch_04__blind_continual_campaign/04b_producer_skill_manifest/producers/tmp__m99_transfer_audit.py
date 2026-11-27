import importlib.util
import json
import pickle
from pathlib import Path
import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root = Path('/holdout_task')
archive = Path('/tmp/hegel_continual_archive')
m = load('candidate', root / 'batch_solver.py')
prior = load('precedent', archive / 'alpha_m99_c04_batch_01_solver.py')
with (root / 'public_batch.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
records = json.loads((archive / 'archive_apparatus_log.json').read_text())
z = np.asarray(records[-1]['facility_telemetry_trace'])[:, [1, 5]]
p = np.asarray([r['facility_telemetry_trace'] for r in records[:-1]])[:, :, [1, 5]]
for label, keep in [('first half', slice(None, 12)), ('second half', slice(12, None))]:
    dist = np.linalg.norm(p[:, keep].mean(1) - z[keep].mean(0), axis=1)
    order = np.argsort(dist)
    print(label, records[order[0]]['completed_campaign_short_name'], 'nearest and runner-up', dist[order[:2]].tolist())
    assert records[order[0]]['completed_campaign_short_name'] == 'alpha_m99_c04_s00'
out = m.solve_batch(d)
bridge = m._fit_bridge(d['calibration_load_cells'])
train, test = d['train'], d['test']
xt, xe = m._features(train, test)
tau, theta = out['estimated_taus'], out['hereditary_weights']
t = d['eval_times']
hold = m._decode(test['hold_channel_raw'], bridge).sum(axis=-1)
curves, rate = prior._chamber_response(hold, theta[2] + xe @ theta[3:], t, tau, theta)
print('archived chamber transfer max strain and rate discrepancy', np.max(np.abs(out['predicted_strain'] - (test['strain_0'][:, None] + curves))), np.max(np.abs(out['initial_rate_pred'] - rate)))
np.testing.assert_allclose(out['predicted_strain'], test['strain_0'][:, None] + curves, rtol=0, atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], rate, rtol=0, atol=1e-12)
reversed_history = dict(d, test=dict(test, past_channel_raw=test['past_channel_raw'][:, ::-1, :]))
changed = m.solve_batch(reversed_history)
for key in out:
    np.testing.assert_allclose(changed[key], out[key], rtol=0, atol=1e-12)
print('deployment outputs unchanged when measured deployment stress histories are reversed')
