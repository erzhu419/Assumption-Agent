import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'viscoplastic_overstress_hereditary_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
design = m._design
for case in pickle.load(open(root + 'public_suc01_C_datasets.pkl', 'rb')):
    d, g = case['dataset'], case['gt']
    train, test = d['train'], d['test']
    h, y = test['stress_hold_0'], d['yield_stress']
    over = (abs(h) > y) | (abs(test['past_stress']) > y).any(axis=1)
    for name, omitted in [('full', []), ('linear-only', [7, 8]), ('no-hold-creep', [7]), ('no-work-surge', [8])]:
        def variant(*args):
            curves, rates = design(*args)
            curves[..., omitted] = 0
            rates[..., omitted] = 0
            return curves, rates
        m._design = variant
        err = np.empty_like(g['te_clean'])
        for lab in np.unique(test['lab_id']):
            tr, te = train['lab_id'] != lab, test['lab_id'] == lab
            fold = dict(d, train={k: v[tr] for k, v in train.items()}, test={k: v[te] for k, v in test.items()})
            out = m.solve_viscoplastic_hereditary_revision(fold)
            err[te] = out['predicted_strain'] - g['te_clean'][te]
        print(name, 'held-out-lab pooled MAE (overall/over/sub):', np.round([abs(err).mean(), abs(err[over]).mean(), abs(err[~over]).mean()], 6))
        for label, mask in [('over negative hold', over & (h < 0)), ('over positive hold', over & (h > 0)), ('sub-yield', ~over)]:
            if mask.any():
                print(' ', label, int(mask.sum()), 'MAE/bias/late MAE:', np.round([abs(err[mask]).mean(), err[mask].mean(), abs(err[mask, -1]).mean()], 6))
        if name == 'full':
            assert np.all(np.isfinite(err))
            assert abs(err).mean() <= .070 and abs(err[over]).mean() <= .080 and abs(err[~over]).mean() <= .065
    m._design = design
