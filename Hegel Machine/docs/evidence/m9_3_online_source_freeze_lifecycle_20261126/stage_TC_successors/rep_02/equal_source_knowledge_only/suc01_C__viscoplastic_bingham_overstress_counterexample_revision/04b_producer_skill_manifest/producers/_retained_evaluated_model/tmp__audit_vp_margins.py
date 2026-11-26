import importlib.util
import pickle
import numpy as np

path = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('vp', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_suc01_C_datasets.pkl', 'rb') as file:
    cases = pickle.load(file)

for i, case in enumerate(cases):
    d, g = case['dataset'], case['gt']
    tr, te = d['train'], d['test']
    sy = d['yield_stress']
    extra = (d['past_dt'], d['eval_times'], sy)
    hot = (np.abs(te['stress_hold_0']) > sy) | (np.abs(te['past_stress']) > sy).any(axis=1)
    xtr, xte, w = m._align(tr, te)
    train_model = m._Response(tr, xtr, *extra)
    test_model = m._Response(te, xte, *extra)
    for name, cols in [('full', np.arange(9)), ('without hold creep', np.r_[0:7, 8]), ('without history surge', np.arange(8))]:
        tau, beta = m._profile(train_model, tr, w, cols)
        a, _ = test_model.design(tau)
        err = np.abs(te['strain_0'][:, None] + a[:, :, cols] @ beta - g['te_clean'])
        print(i, name, 'sub-yield/overstressed MAE', np.round([err[~hot].mean(), err[hot].mean()], 5))
    for label in np.unique(te['lab_id']):
        fit = tr['lab_id'] != label
        target = te['lab_id'] == label
        fold = {k: v[fit] for k, v in tr.items()}
        xt, xv, weights = m._align(fold, te)
        for name, weight in [('cell-balanced', weights), ('unweighted', np.ones_like(weights))]:
            tau, beta = m._profile(m._Response(fold, xt, *extra), fold, weight)
            a, _ = m._Response(te, xv, *extra).design(tau)
            err = np.abs(te['strain_0'][:, None] + a @ beta - g['te_clean'])
            scores = [round(err[mask].mean(), 5) for mask in (target, target & ~hot, target & hot) if mask.any()]
            vt = np.linalg.norm(beta[-2:] - g['vp_theta']) / np.linalg.norm(g['vp_theta'])
            print(i, 'held lab', label, name, 'all/sub-yield/overstressed MAE', scores, 'max tau / VP relative errors', np.round([max(abs(tau / g['taus'] - 1)), vt], 5))
    # The public rows do not separate the two activations. Recombine public
    # histories and hold stresses, including exact yield, to test them independently.
    h = te['past_stress']
    h_ok = h[~(np.abs(h) > sy).any(axis=1)][0]
    h_hot = h[(np.abs(h) > sy).any(axis=1)][0]
    u_hot = te['stress_hold_0'][te['stress_hold_0'] < -sy][0]
    u = np.array([sy, u_hot, -sy, u_hot])
    probe = {'past_stress': np.array([h_ok, h_ok, h_hot, h_hot]), 'stress_hold_0': u}
    a, rate = m._Response(probe, np.zeros((4, 4)), *extra).design(np.array([0.58, 3.9]))
    expected = np.array([[False, False], [True, False], [False, True], [True, True]])
    np.testing.assert_equal(rate[:, -2:] != 0, expected)
    np.testing.assert_equal(np.any(a[:, :, -2:] != 0, axis=1), expected)
    assert rate[1, -2] < 0 and rate[2, -1] < 0
    print(i, 'independent yield activation, exact yield boundary, and negative signs passed')
