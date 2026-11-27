import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c03_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_batch_06.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']
p,h,x,o,dt,t = m._prepare(d)
tr = d['train']; lab = tr['lab_id']; n = len(lab)
y = tr['strain_curves_obs']-o[:n,None]; r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
rmse = lambda v: np.sqrt(np.nanmean(np.square(v)))
def mad(v):
    return max(1e-6, 1.4826*np.nanmedian(np.abs(v-np.nanmedian(v))))
def fits(rows):
    args = p[:n][rows],h[:n][rows],x[:n][rows],dt,t
    yy,rr = y[rows],r[rows]
    f,w = m._fit(*args,yy,rr)
    c,_ = m._design(f,*args); err = c@w-yy
    center,scale = np.nanmedian(err),mad(err)
    q = np.abs(err-center)>6*scale
    return (f,w),m._fit(*args,np.where(q,np.nan,yy),rr),center,scale,int(q.sum())
metrics = [[],[]]
for j in range(4):
    a,b,center,scale,ncut = fits((lab!=j)&complete)
    rows = (lab==j)&complete
    for k,(f,w) in enumerate((a,b)):
        c,v = m._design(f,p[:n][rows],h[:n][rows],x[:n][rows],dt,t)
        e = c@w-y[rows]; er = v@w-r[rows]
        # Also report the error in the central 90% of residuals, so isolated
        # spikes in validation cannot dominate the comparison.
        cutoff = np.nanquantile(np.abs(e),.90)
        vals = (rmse(e[np.abs(e)<=cutoff]),np.nanmedian(np.abs(e)),rmse(er))
        metrics[k].append(vals)
        print(j,k,'central strain RMSE / median abs strain / rate RMSE',np.round(vals,6),'training spikes removed by challenger',ncut,'tau',np.round(np.exp(f),5))
print('worst observed lab by method',np.max(metrics,axis=1))
a,b,_,_,ncut = fits(complete)
g = item['gt']; missing = ~np.isfinite(d['test']['coupon_features_raw']).all(axis=1)
for k,(f,w) in enumerate((a,b)):
    c,v = m._design(f,p[n:],h[n:],x[n:],dt,t); e = o[n:,None]+c@w-g['te_clean']; er = v@w-g['te_true_rate0']; print('public check method',k,'complete/missing RMSE',[(rmse(e[q]),rmse(er[q])) for q in (~missing,missing)],'relative tau max',max(abs(np.exp(f)/g['taus']-1)),'training spikes removed by challenger',ncut)
