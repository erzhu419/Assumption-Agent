import importlib.util
import json
from pathlib import Path
import pickle
import numpy as np

root = Path('/holdout_task')
name = 'alpha_w04_batch_08_solver.py'
def load(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def rmse(x):
    return float(np.sqrt(np.mean(x**2)))

def subset(state, mask):
    return state[:2] + tuple(v[mask] for v in state[2:])

base = load(next(Path('/tmp/.hegel_candidate_archive/v01_9ab5ef51').rglob(name)), 'baseline')
cand = load(root / name, 'candidate')
with (root / 'public_alpha_w04_batch_08.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
t, dt, sy = (d[k] for k in ('eval_times', 'past_dt', 'yield_stress'))
panel = d['calibration_load_cells']
old, new = base._calibrate(panel), cand._calibrate(panel)
x, xt = cand._align(tr, te)
st = cand._prepare(dt, *cand._stress(tr, new), x, sy)
st0 = cand._prepare(dt, *cand._stress(tr, old), x, sy)
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
yielded = (st[-1] != 0) | (st[-2] != 0)
train = complete & (lab != 1)
val = complete & (lab == 1) & yielded
te_val = (te['lab_id'] == 1) & np.isfinite(te['coupon_features_raw']).all(axis=1)
taus, w = cand._fit(t, subset(st, train), y[train], r[train])
metrics = {}
for tag, affine, state in [('baseline', old, st0), ('candidate', new, st)]:
    a, b = cand._design(taus, t, subset(state, val))
    tst = cand._prepare(dt, *cand._stress(te, affine), xt, sy)
    _, onset = cand._design(taus, t, subset(tst, te_val))
    metrics[tag] = {'yielded_curve_rmse': rmse(a @ w - y[val]), 'yielded_onset_rmse': rmse(b @ w - r[val]), 'test_onset_rmse': rmse(onset @ w - te['instant_rate_obs'][te_val])}
print(json.dumps({'nogood_id': 'ng_t06', 'risk_kind': 'cross-wing calibration transfer', 'held_out_lab': 1, 'yielded_coupon_count': int(val.sum()), 'test_onset_coupon_count': int(te_val.sum()), 'same_fitted_constitutive_parameters': True, 'metrics': metrics}), flush=True)
assert all(metrics['candidate'][k] < metrics['baseline'][k] for k in metrics['baseline']), metrics
assert np.array_equal(new[2:], old[2:]), (old[2:], new[2:])
cmd, obs = cand._rows(panel, 0)
spot_max = float(np.max(abs(new[0,0]*cmd + new[0,1] - obs)))
print(json.dumps({'wing_b_calibration_unchanged': True, 'lab_0_max_spot_residual': spot_max, 'lab_0_consensus_cutoff': 0.003}))
assert spot_max < 0.003, spot_max
