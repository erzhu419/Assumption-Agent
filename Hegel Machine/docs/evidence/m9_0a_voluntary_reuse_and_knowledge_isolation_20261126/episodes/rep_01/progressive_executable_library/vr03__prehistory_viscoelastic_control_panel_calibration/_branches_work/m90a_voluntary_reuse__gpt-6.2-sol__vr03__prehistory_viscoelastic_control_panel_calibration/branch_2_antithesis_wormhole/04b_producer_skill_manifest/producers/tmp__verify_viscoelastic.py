import importlib.util
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

path = '/holdout_task/viscoelastic_panel_calibration.py'
spec = importlib.util.spec_from_file_location('viscoelastic_panel_calibration', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
run = solver.calibrate_viscoelastic_cohort

# A constant history followed by a step gives a closed-form memory trajectory,
# including the gap between the prehistory and observation windows.
taus = np.array([0.41, 1.67])
t = np.cumsum(np.linspace(0.06, 0.17, 36)) + 0.29
u = np.full(t.size, 2.8)
tr = dict(past_timestamps=np.array([-1.52, -0.64, 0.0]),
          past_strain_history=np.full(3, -1.2), timestamps=t, strain_u=u)
expected = 2.8 - 4.0*np.exp(-t[:, None]/taus)
np.testing.assert_allclose(solver._stress_design(tr, taus)[:, 2:], expected, atol=2e-14, rtol=0)

with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as handle:
    data = pickle.load(handle)
worst = dict(auc=1.0, brier=0.0, mae=0.0, cell_mae=0.0, loco_auc=1.0, loco_brier=0.0)
for k, item in enumerate(data):
    cohort, gt = item['cohort'], item['gt']
    train, query = cohort['train'], cohort['query']
    out = run(cohort)
    assert set(out) == {'aligned_query_moduli', 'query_probabilities', 'modulus_weights'}
    z, p, beta = (out[key] for key in ('aligned_query_moduli', 'query_probabilities', 'modulus_weights'))
    n = len(query['center_id'])
    for arr, shape in [(z, (n, 3)), (p, (n,)), (beta, (3,))]:
        assert isinstance(arr, np.ndarray) and arr.shape == shape and arr.dtype.kind == 'f'
        assert np.isfinite(arr).all()
    assert np.all((p > 0) & (p < 1))
    mask = query['missing_channel_mask']
    train_z = solver._aligned_moduli(train, np.asarray(cohort['taus']))
    by_cell = [(train['center_id'] == c) & (train['stratum'] == s)
               for c in np.unique(train['center_id']) for s in np.unique(train['stratum'])]
    means_of_sums = np.sum([np.nansum(train_z[i], axis=0)/i.sum() for i in by_cell], axis=0)
    means_of_counts = np.sum([np.isfinite(train_z[i]).mean(axis=0) for i in by_cell], axis=0)
    expected_fill = means_of_sums/means_of_counts
    np.testing.assert_allclose(z[mask], np.broadcast_to(expected_fill, z.shape)[mask], atol=1e-12, rtol=0)
    for c in np.unique(query['center_id']):
        rows = query['center_id'] == c
        y = gt['y_query'][rows]
        auc = roc_auc_score(y, p[rows])
        brier = np.mean((p[rows]-y)**2)
        mae = np.mean(np.abs(z[rows]-gt['Z_query'][rows]))
        cell_mae = max(np.mean(np.abs(z[i]-gt['Z_query'][i])) for s in np.unique(query['stratum'][rows]) for i in [rows & (query['stratum'] == s)])
        print(f'public={k} center={c} AUROC={auc:.5f} Brier={brier:.5f} MAE={mae:.5f} max_stratum_MAE={cell_mae:.5f}')
        worst['auc'] = min(worst['auc'], auc)
        for name, value in [('brier', brier), ('mae', mae), ('cell_mae', cell_mae)]:
            worst[name] = max(worst[name], value)
    print(f'public={k} fitted slopes={beta.round(4)} reference={gt["beta"]}')
    def take(part, rows):
        return {key: [val[i] for i in np.flatnonzero(rows)] if key == 'traces' else val[rows] for key, val in part.items()}
    for c in np.unique(train['center_id']):
        test = train['center_id'] == c
        fold_query = take(train, test)
        y = fold_query.pop('y')
        pred = run(dict(taus=cohort['taus'], train=take(train, ~test), query=fold_query))['query_probabilities']
        auc, brier = roc_auc_score(y, pred), np.mean((pred-y)**2)
        worst['loco_auc'] = min(worst['loco_auc'], auc)
        worst['loco_brier'] = max(worst['loco_brier'], brier)
        print(f'public={k} training LOCO center={c} AUROC={auc:.5f} Brier={brier:.5f}')
print('Worst slices:', {k: round(v, 6) for k, v in worst.items()})
assert worst['auc'] >= 0.88 and worst['brier'] <= 0.160 and worst['mae'] <= 0.22
print('Verified target artifact with python3 -I')
