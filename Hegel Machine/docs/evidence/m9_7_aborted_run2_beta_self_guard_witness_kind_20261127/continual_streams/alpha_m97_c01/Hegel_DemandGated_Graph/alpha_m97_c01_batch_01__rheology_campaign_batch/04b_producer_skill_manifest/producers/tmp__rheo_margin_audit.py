import importlib.util, json, pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
s = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c01_batch_01_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; lab = c['lab_id']; cmd = c['cmd_stress']; true = c['true_stress']
ref = np.array([np.polyfit(cmd[lab==j], true[lab==j], 1) for j in range(4)])
np.testing.assert_allclose(m._calibrate(d), ref, atol=1e-12)
worst_stress = 0.
for j in range(4):
    ids = np.flatnonzero(lab == j)
    for sign in (-1, 1):
        keep = np.concatenate((np.flatnonzero(lab != j), ids[:5]))
        pert = {k: np.asarray(v)[keep].copy() for k,v in c.items()}
        pert['true_stress'][-1] += sign * 4.
        expected = np.polyfit(cmd[ids[:4]], true[ids[:4]], 1)
        est = m._calibrate(dict(d, calibration_load_cells=pert))
        np.testing.assert_allclose(est[j], expected, atol=1e-12)
        others = np.arange(4) != j
        np.testing.assert_allclose(est[others], ref[others], atol=1e-12)
        for split in ('train','test'):
            data = d[split]; q = data['lab_id'] == j
            vals = np.r_[data['past_stress_cmd'][q].ravel(), data['stress_hold_cmd'][q]]
            worst_stress = max(worst_stress, np.max(abs(np.polyval(est[j]-ref[j], vals))))
print('Four clean calibration points plus either spike sign, worst stress shift from full sweep:', worst_stress)
with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
    history = json.load(f)
for j in range(4):
    idx = np.flatnonzero(lab == j)
    keep = (lab != j); keep[idx[0]] = True
    sparse = {k: np.asarray(v)[keep] for k,v in c.items()}
    sweep = next(h for h in reversed(history) if np.count_nonzero(np.asarray(h['lab_id']) == j) >= 2)
    q = np.asarray(sweep['lab_id']) == j
    expected = np.polyfit(np.asarray(sweep['cmd_stress'])[q], np.asarray(sweep['true_stress'])[q], 1)
    np.testing.assert_allclose(m._calibrate(dict(d, calibration_load_cells=sparse))[j], expected, atol=1e-12)
print('One-point history fallback verified for each channel')
past,hold,x,s0 = m._inputs(d); tr = d['train']; n = len(tr['lab_id']); y = np.c_[tr['strain_curves_obs']-s0[:n,None], tr['instant_rate_obs']]
active = m._Model(d,past[:n],hold[:n],x[:n]).active
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
r = np.empty_like(y)
for j in range(4):
    te = tr['lab_id']==j; ti = ~te
    def model(mask):
        return m._Model(d,past[:n][mask],hold[:n][mask],x[:n][mask])
    tau,w = m._fit(model(ti),y[ti]); r[te] = model(te).design(np.log(tau))[0] @ w-y[te]
slices = [(tr['lab_id']==j)&(active==gate)&complete for j in range(4) for gate in (False,True)]
print('Complete-coupon lab-by-activation worst absolute signed median bias, each horizon then rate:', np.max([abs(np.median(r[q],axis=0)) for q in slices],axis=0).round(5))
