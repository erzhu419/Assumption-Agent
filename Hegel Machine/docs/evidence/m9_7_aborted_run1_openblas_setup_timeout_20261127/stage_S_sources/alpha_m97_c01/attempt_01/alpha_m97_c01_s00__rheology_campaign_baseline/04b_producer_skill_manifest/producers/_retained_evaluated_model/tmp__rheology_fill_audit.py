import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m97_c01_s00.pkl','rb'))[0]['dataset']
tr = d['train']
raw, panel, lab, stratum = [np.concatenate([d[s][k] for s in ('train','test')]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')]
aligned, n = m._prepare(d)
past, hold, reference = aligned
x = raw.copy()
for l in np.unique(lab):
    sel = lab==l
    p = panel[sel]
    p = p[:,np.isfinite(p).sum(axis=0)>=2]
    x[sel] = (x[sel]-np.median(np.nanmean(p,axis=0)))/np.median(np.nanstd(p,axis=0))
keys, cell = np.unique(np.c_[lab,stratum],axis=0,return_inverse=True)
def fill(x, method):
    means = np.array([np.nanmean(x[cell==j],axis=0) for j in range(len(keys))])
    equal = np.nanmean(means,axis=0)
    if method == 'row_weight':
        counts=np.bincount(cell)
        return np.nansum(x/counts[cell,None],axis=0)/np.sum(np.isfinite(x)/counts[cell,None],axis=0)
    if method == 'equal_observed_cell':
        return equal
    return np.where(np.isfinite(means[cell]),means[cell],equal)
np.testing.assert_allclose(reference,np.where(np.isfinite(x),x,fill(x,'row_weight')))
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
dt,t = d['past_dt'],d['eval_times']
missing = np.isnan(x[:n]).any(axis=1)
methods=('row_weight','equal_observed_cell','local_cell')
scores={method:[] for method in methods}
for l in np.unique(lab[:n]):
    keep = lab[:n]!=l
    h = ~keep
    # Reconstruct one masked feature at a time on held-lab complete rows.
    # The actual missing-row evaluation below also checks the fit end to end.
    complete = np.flatnonzero(h & ~missing)
    for method in methods:
        a = np.where(np.isfinite(x),x,fill(x,method))
        data = (past[:n],hold[:n],a[:n])
        logs,w=m._fit(tuple(v[keep] for v in data),dt,t,y[keep],r[keep])
        c,b=m._predict(logs,w,tuple(v[h] for v in data),dt,t)
        e,er=np.abs(c-y[h]),np.abs(b-r[h])
        synthetic=[]
        for j in range(x.shape[1]):
            for i in complete:
                masked=x.copy()
                masked[i,j]=np.nan
                f=fill(masked,method)
                synthetic.append((f[j] if f.ndim==1 else f[i,j])-x[i,j])
        row=[np.median(e,axis=0).max(),np.quantile(er,.9),np.median(e[missing[h]]),np.median(er[missing[h]]),np.sqrt(np.mean(np.square(synthetic)))]
        scores[method].append(row)
        print('lab',l,method,np.round(row,6))
for method,rows in scores.items():
    print('WORST',method,np.round(np.max(rows,axis=0),6))
