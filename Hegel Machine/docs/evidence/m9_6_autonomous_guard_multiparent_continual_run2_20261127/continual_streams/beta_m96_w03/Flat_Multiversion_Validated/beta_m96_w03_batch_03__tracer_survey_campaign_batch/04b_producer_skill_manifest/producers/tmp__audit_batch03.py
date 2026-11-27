import importlib.util, json, pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'beta_m96_w03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.loads((root / 'public_beta_m96_w03_batch_03.pkl').read_bytes())[0]
s, g = item['study'], item['gt']
A, T = m._forward(s)
z = np.asarray(s['Z_nuisance'])
r = m._calibrated_observations(s)
D = np.column_stack((A, r))
independent = D - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ D)
P, y = m._profile(s, A, r)
print('Independent nuisance projection max error:', np.max(np.abs(np.column_stack((P,y)) - independent)))
assert np.max(np.abs(np.column_stack((P,y)) - independent)) < 1e-12
assert np.max(np.abs(P - g['S_tilde'])) < 1e-12
out = m.solve_beta_m96_w03_batch_03(s)
for i, (budget, w, cert) in enumerate(zip(s['budget_grid'], out['weights_by_budget'], out['kkt_certificates'])):
    err = independent[:, :-1] @ w - independent[:, -1]
    grad = independent[:, :-1].T @ err + s['source_ridge'] * w
    cap = s['u_box']
    stationarity = np.max(np.abs(grad - cert['nu_sum'] - cert['lam_lower'] + cert['lam_upper']))
    complementarity = max(np.max(abs(w * cert['lam_lower'])), np.max(abs((cap-w) * cert['lam_upper'])))
    gap = cert['primal_objective'] - cert['dual_lower_bound']
    print('budget', i, 'sum error / stationarity / complementarity / gap:', abs(w.sum()-budget), stationarity, complementarity, gap)
    assert np.all(w >= -1e-12) and np.all(w <= cap + 1e-12)
    assert max(abs(w.sum()-budget), stationarity, complementarity, abs(gap)) < 1e-10
assert np.linalg.norm(out['terminal_field_mid_budget'] - g['terminal_field_mid_budget']) / np.linalg.norm(g['terminal_field_mid_budget']) < 1e-6

archive = json.loads(m._HISTORY.read_text())
for i, episode in enumerate(archive + [s['sensor_calibration_sweeps']]):
    # Only prior episodes may supply calibration; the held-out readings are scored, not fitted.
    snapshot = Path('/tmp/batch03_calibration_audit_prefix.json')
    snapshot.write_text(json.dumps(archive[:i]))
    m._HISTORY = snapshot
    bank, x, raw = (np.asarray(episode[k]) for k in ('bank_id', 'ref_signal', 'raw_readout'))
    full = m._calibration_coefficients(episode)
    for kept in (0, 1, 2):
        worst = np.zeros(2)
        for b in range(4):
            idx = np.flatnonzero(bank == b)
            for selected in (idx[:kept], idx[-kept:] if kept else idx[:0]):
                use = bank != b
                use[selected] = True
                training = dict(zip(('bank_id','ref_signal','raw_readout'), (bank[use],x[use],raw[use])))
                gain, offset = m._calibration_coefficients(training)[b]
                held = (bank == b) & ~use
                physical_error = (raw[held] - offset) / gain - x[held]
                baseline_error = (raw[held] - full[b,1]) / full[b,0] - x[held]
                worst = np.maximum(worst, [np.max(abs(physical_error)), np.max(abs(baseline_error))])
        print('prefix size', i, 'kept', kept, 'worst held-out error / full-sweep residual:', *worst)
