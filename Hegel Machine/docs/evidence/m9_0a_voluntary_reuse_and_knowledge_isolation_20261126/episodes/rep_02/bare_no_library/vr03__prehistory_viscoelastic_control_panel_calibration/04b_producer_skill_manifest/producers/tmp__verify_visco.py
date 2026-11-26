import importlib.util
import pickle
import numpy as np
from scipy.stats import rankdata

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('viscoelastic_panel_calibration', p+'viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(p+'public_viscoelastic_cohort.pkl', 'rb') as f:
    cases = pickle.load(f)

def auc(y, v):
    n = int(y.sum())
    return (rankdata(v)[y == 1].sum() - n*(n+1)/2)/(n*(len(y)-n))

rng = np.random.default_rng(821)
errors = []
for i, case in enumerate(cases):
    co, g = case['cohort'], case['gt']
    tr, q = co['train'], co['query']
    r = m.calibrate_viscoelastic_cohort(co)
    assert set(r) == {'aligned_query_moduli', 'query_probabilities', 'modulus_weights'}
    x, v, b = (r[k] for k in ('aligned_query_moduli','query_probabilities','modulus_weights'))
    y = np.asarray(g['y_query'])
    n = len(y)
    assert x.shape == (n,3) and v.shape == (n,) and b.shape == (3,)
    assert all(a.dtype.kind == 'f' and np.isfinite(a).all() for a in (x,v,b))
    assert ((v > 0) & (v < 1)).all()
    print('cohort',i,'weight error',np.round(b-g['beta'],4))
    for c in np.unique(q['center_id']):
        w = q['center_id'] == c
        metrics = np.abs(x[w]-g['Z_query'][w]).mean(),auc(y[w],v[w]),np.mean((v[w]-y[w])**2)
        print('query center',c,'MAE/AUROC/Brier',np.round(metrics,5))
        assert metrics[0] <= .22 and metrics[1] >= .88 and metrics[2] <= .16
    mask = np.asarray(q['missing_channel_mask'])
    xt = m._aligned(tr, np.asarray(co['taus']))
    groups = [(tr['center_id']==c)&(tr['stratum']==s) for c in np.unique(tr['center_id']) for s in range(3)]
    expected = np.nanmean([np.nanmean(xt[g],axis=0) for g in groups],axis=0)
    np.testing.assert_allclose(x[mask],np.broadcast_to(expected,x.shape)[mask])
    taus = np.asarray(co['taus'])
    subset_rng = np.random.default_rng(185)
    subsets = np.array([subset_rng.choice(36,4,replace=False) for _ in range(96)])
    for trace in q['traces']:
        tp = np.asarray(trace['past_timestamps']); up = np.asarray(trace['past_strain_history'])
        t = np.asarray(trace['timestamps']); u = np.asarray(trace['strain_u'])
        # Independent direct convolution using right-endpoint-held strain.
        ts = np.r_[tp,t]; us = np.r_[up,u]; modes = []
        for tau in taus:
            z = []
            for k in range(len(tp),len(ts)):
                z.append(us[0]*np.exp((ts[0]-ts[k])/tau) + us[1:k+1]@(np.exp((ts[1:k+1]-ts[k])/tau)-np.exp((ts[:k]-ts[k])/tau)))
            modes.append(z)
        a = np.column_stack((np.ones(len(t)),u,*modes))
        actual = m._design(trace,taus)
        np.testing.assert_allclose(actual,a,rtol=1e-11,atol=1e-12)
        for count in (4,6):
            coeff = rng.normal(size=4)
            signal = a@coeff
            j = rng.choice(len(t),count,replace=False)
            signal[j] += rng.choice([-1,1],size=count)*rng.uniform(4.5,9.,size=count)
            errors.append(np.max(np.abs(m._robust_fit(actual,signal,subsets)-coeff[1:])))
print('known coefficient replay: fits',len(errors),'worst absolute error',max(errors))
assert max(errors) < 1e-6
