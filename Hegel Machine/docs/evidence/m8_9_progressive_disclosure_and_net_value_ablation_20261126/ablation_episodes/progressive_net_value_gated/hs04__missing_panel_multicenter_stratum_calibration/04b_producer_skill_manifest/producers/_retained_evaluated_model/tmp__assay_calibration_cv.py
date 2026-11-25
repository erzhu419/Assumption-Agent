import importlib.util
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_multicenter_assay.pkl','rb') as handle:
    cohorts = pickle.load(handle)['cohorts']

def fit(x, y, s, c, z, v, C, center_scale):
    strata = np.unique(s)
    ct = (c[:,None] == np.unique(c)).astype(float)
    ct -= ct.mean(0)
    train = np.column_stack((x, s[:,None] == strata, center_scale * ct))
    test = np.column_stack((z, v[:,None] == strata, np.zeros((len(z), ct.shape[1]))))
    model = LogisticRegression(C=C, max_iter=1000).fit(train, y)
    return model.predict_proba(test)[:,1]

settings = [(C, a) for C in (0.1, 0.3, 1.0, 3.0) for a in (0.0, 0.25, 1.0)]
all_oof = {setting: [] for setting in settings}
for i, co in enumerate(cohorts):
    d = co['dataset']
    t = d['train']
    x = m._align(t, *m._reference_transforms(d['reference_controls']))
    y, s, c = (t[k] for k in ('y','stratum','center_id'))
    centers = np.unique(c)
    def loo(available, setting):
        entries = []
        for center in available:
            test = c == center
            train = np.isin(c, available) & ~test
            p = fit(x[train],y[train],s[train],c[train],x[test],s[test],*setting)
            entries.append((roc_auc_score(y[test],p),brier_score_loss(y[test],p),log_loss(y[test],p)))
        return entries
    for setting in settings:
        all_oof[setting].extend(loo(centers, setting))
    nested = []
    for center in centers:
        remaining = centers[centers != center]
        best = min(settings, key=lambda setting: np.mean(loo(remaining,setting),axis=0)[2])
        train = c != center
        p = fit(x[train],y[train],s[train],c[train],x[~train],s[~train],*best)
        nested.append((best, round(roc_auc_score(y[~train],p),3), round(brier_score_loss(y[~train],p),3)))
    print('cohort',i,'nested outer (chosen C, center scale), AUROC, Brier:',nested)
for setting, results in all_oof.items():
    z = np.array(results)
    print(setting, 'mean AUROC/Brier/logloss', np.round(z.mean(0),4), 'worst AUROC/Brier', np.round([z[:,0].min(),z[:,1].max()],4))
