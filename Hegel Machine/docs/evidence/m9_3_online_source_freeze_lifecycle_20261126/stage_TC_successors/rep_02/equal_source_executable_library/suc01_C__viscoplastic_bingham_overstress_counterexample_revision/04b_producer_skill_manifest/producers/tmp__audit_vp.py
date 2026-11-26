import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'viscoplastic_overstress_hereditary_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for case in pickle.load(open(root + 'public_suc01_C_datasets.pkl', 'rb')):
    d, g = case['dataset'], case['gt']
    train, test = d['train'], d['test']
    dt, t = d['past_dt'], d['eval_times']
    y, tau = float(d['yield_stress']), np.asarray(g['taus'])
    s, h = test['past_stress'], test['stress_hold_0']
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    work = np.sum(np.maximum(np.abs(s) - y, 0.)**2 * dt, axis=1)
    ov = np.sign(h) * np.maximum(np.abs(h) - y, 0.)**2
    over = (work > 0) | (ov != 0)
    state = np.zeros((len(h), 2))
    for j, step in enumerate(dt):
        state = state * np.exp(-step / tau) + s[:, j, None] * (-np.expm1(-step / tau))
    x = g['te_Z']
    vals = m._design_inputs(test, x, dt, y)
    A, R = m._design(vals, dt, age, t, tau)
    gaps = h[:, None] - state
    assert np.allclose(R[:, :2] * tau, gaps)
    assert np.allclose(R[:, 7], ov) and np.allclose(R[:, 8] * tau[0], work*h)
    assert not np.any(A[~over][:, :, 7:9])
    assert np.allclose(A[:, :, 7], ov[:, None]*t)
    assert np.allclose(A[:, :, 8], (work*h)[:, None]*(-np.expm1(-t/tau[0])))
    c = np.linalg.lstsq(A.reshape(-1, 9), (g['te_clean']-test['strain_0'][:, None]).ravel(), rcond=None)[0]
    closure = np.max(np.abs(m._evaluate(A,c) + test['strain_0'][:, None] - g['te_clean']))
    rate_closure = np.max(np.abs(R@c - g['te_true_rate0']))
    print('oracle clean-curve and rate closure:', closure, rate_closure)
    for lab in np.unique(train['lab_id']):
        tr = train['lab_id'] != lab
        te = test['lab_id'] == lab
        if not np.any(te):
            continue
        fold = dict(d, train={k: v[tr] for k, v in train.items()}, test={k: v[te] for k, v in test.items()})
        p = m.solve_viscoplastic_hereditary_revision(fold)
        err = p['predicted_strain'] - g['te_clean'][te]
        print('held-out lab', lab, 'max tau rel', round(np.max(np.abs(p['estimated_taus']/tau-1)), 5), 'vp rel L2', round(np.linalg.norm(p['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta']), 5))
        for regime, mask in [('all', np.ones(te.sum(), bool)), ('over', over[te]), ('sub', ~over[te])]:
            if mask.any():
                print(' ', regime, 'n', mask.sum(), 'MAE', round(np.abs(err[mask]).mean(), 6), 'signed bias', round(err[mask].mean(), 6))
        print('  stratum MAEs', [(int(j), round(np.abs(err[fold['test']['stratum']==j]).mean(), 6)) for j in np.unique(fold['test']['stratum'])])
