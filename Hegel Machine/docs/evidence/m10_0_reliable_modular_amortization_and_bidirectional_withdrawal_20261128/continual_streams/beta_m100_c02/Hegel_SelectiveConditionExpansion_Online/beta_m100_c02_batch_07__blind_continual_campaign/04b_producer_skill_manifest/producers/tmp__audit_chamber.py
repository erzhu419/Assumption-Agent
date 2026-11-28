import json
import pickle
import sys
from pathlib import Path

import numpy as np

root = Path('/holdout_task')
sys.path.insert(0, str(root))
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import _axis, _responses, solve_downstream

item = pickle.loads(root.joinpath('public_batch.pkl').read_bytes())[0]
s = item['study']
payload = json.loads(root.joinpath('_upstream_param_payload.json').read_text())
S, fields = _responses(s)
z = s['Z_nuisance']
q, _ = np.linalg.qr(z, mode='reduced')
ns = len(s['sensor_ij'])
nsh = len(s['mu_shifts'])
assert ns % 4 == 0
row_err = max(np.max(np.abs(_axis(s['kappa_'+axis], s['vel_'+axis], s['L'+axis]/(len(s['kappa_'+axis])-1)) @ np.ones(len(s['kappa_'+axis])))) for axis in ('x', 'y'))
assert row_err < 1e-11
print('Constant-field transport residual:', row_err)
for label, params in [('study-local sweeps', None), ('staged payload', payload)]:
    raw_slices, profiled_slices = [], []
    for p in item['campaign_diagnostics']:
        y = calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']), params)['y_obs_raw']
        r = S @ p['w_pulse'] - y
        raw_slices.append(np.sqrt(np.mean(r.reshape(nsh, ns//4, 4)**2, axis=1)))
        profiled_slices.append(np.sqrt(np.mean((r-q@(q.T@r)).reshape(nsh, ns//4, 4)**2, axis=1)))
    raw = np.max(raw_slices, axis=(0, 1))
    profiled = np.max(profiled_slices, axis=(0, 1))
    print(label, 'worst RMSE per bank across shifts and pulses:', raw, 'after nuisance projection:', profiled)
    if params is None:
        assert raw.max() < 2e-6
out = solve_batch(s)
w = out['weights']
cal = calibrate_upstream(s)
assert np.array_equal(w, solve_downstream(cal)['weights'])
assert np.allclose(out['terminal_field'], np.einsum('m,mij->ij', w, fields), rtol=0, atol=1e-13)
P = np.eye(len(S)) - z @ np.linalg.solve(z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T)
r = P @ (S @ w - cal['y_obs_raw'])
g = (P@S).T @ r + s['source_ridge']*w
cert = out['kkt_certificate']
lo, hi = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
cap = s['u_box']
errors = [abs(w.sum()-s['total_budget']), max(0., -w.min(), w.max()-cap), np.max(np.abs(g-cert['nu_sum']-lo+hi)), np.max(np.abs(lo*w)), np.max(np.abs(hi*(cap-w))), abs(cert['primal_objective']-cert['dual_lower_bound'])]
print('Composed public candidate: max feasibility/KKT error:', max(errors))
assert max(errors) < 1e-10
