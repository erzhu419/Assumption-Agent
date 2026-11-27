import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'alpha_m97_c03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c03_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d, gt = item['dataset'], item['gt']
    tr, te = d['train'], d['test']
    print('dataset fields', sorted(d), 'coupon fields', sorted(te))
    c = d['calibration_load_cells']
    print('current calibration rows by lab', np.bincount(c['lab_id'], minlength=4))
    print('calibration command range', np.min(c['cmd_stress']), np.max(c['cmd_stress']))
    out = m.solve_alpha_m97_c03_batch_07(d)
    tau = out['estimated_taus']
    w = np.r_[out['hereditary_weights'], out['regime_mode_weights']]
    _, x = m._align(tr, te)
    cal = m._calibrate(c)
    gain, bias = cal[te['lab_id']].T
    past = te['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = te['stress_hold_cmd'] * gain + bias
    z = np.zeros((len(hold), 2))
    damage = np.zeros(len(hold))
    for s, dt in enumerate(d['past_dt']):
        decay = np.exp(-dt / tau)
        z = decay * z + (1 - decay) * past[:, s, None]
        damage += np.maximum(0, np.abs(past[:, s]) - d['crit_stress'])**2 * dt
    gap = hold[:, None] - z
    t = d['eval_times']
    for band in (d['shear_banding_index'], 0.50, 0.500001):
        act = np.tanh(damage) if band <= 0.5 else np.zeros_like(damage)
        drift = w[2] + x @ w[3:7] + w[8] * act * z[:, 0]
        prony = (gap * w[:2]) @ (1 - np.exp(-t[:, None] / tau)).T
        surge = (w[7] * act * hold)[:, None] * (1 - np.exp(-t / tau[0]))
        explicit = prony + surge + drift[:, None] * (np.log1p(t) if band <= 0.5 else t)
        initial = (gap * w[:2] / tau).sum(1) + drift + w[7] * act * hold / tau[0]
        C, V = m._design(te, x, cal, d['past_dt'], t, d['crit_stress'], band)(tau)
        np.testing.assert_allclose(C @ w, explicit, atol=1e-12, rtol=0)
        np.testing.assert_allclose(V @ w, initial, atol=1e-12, rtol=0)
    # Look for a signed lab effect that pooled RMSE could hide. Missing-feature
    # coupons are a separate slice, not evidence of a lab offset.
    observed = np.isfinite(te['coupon_features_raw']).all(1)
    ce = out['predicted_strain'] - gt['te_clean']
    re = out['initial_rate_pred'] - gt['te_true_rate0']
    for lab in np.unique(te['lab_id']):
        k = observed & (te['lab_id'] == lab)
        print('complete lab', lab, int(k.sum()), 'signed terminal/rate bias', np.mean(ce[k, -1]), np.mean(re[k]), 'terminal/rate RMSE', np.sqrt(np.mean(ce[k, -1]**2)), np.sqrt(np.mean(re[k]**2)))
    print('explicit memory and curve/rate identities passed at the measured band and both threshold sides')
