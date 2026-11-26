import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from viscoelastic_panel_calibration import calibrate_viscoelastic_cohort
for i, item in enumerate(pickle.load(open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb'))):
    co, gt = item['cohort'], item['gt']
    out = calibrate_viscoelastic_cohort(co)
    z, p, b = (out[k] for k in ('aligned_query_moduli','query_probabilities','modulus_weights'))
    y, c = gt['y_query'], co['query']['center_id']
    n = len(y)
    assert z.shape==(n,3) and p.shape==(n,) and b.shape==(3,)
    assert np.isfinite(z).all() and np.isfinite(b).all() and ((p>0)&(p<1)).all()
    print(i, 'MAE', np.abs(z-gt['Z_query']).mean(), 'Brier', np.mean((y-p)**2), 'AUC', [roc_auc_score(y[c==j],p[c==j]) for j in np.unique(c)], 'beta', b)
    for j in np.unique(c):
        for m in [~co['query']['missing_channel_mask'], co['query']['missing_channel_mask']]:
            print(j,int(m.sum()),np.round(np.mean(np.abs(z[c==j][m[c==j]]-gt['Z_query'][c==j][m[c==j]])),4),end='; ')
    print()
