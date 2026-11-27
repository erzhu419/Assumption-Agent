import importlib.util, pickle, numpy as np
from pathlib import Path
p = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', p/'alpha_m97_c01_batch_04_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d = pickle.load(open(p/'public_alpha_m97_c01_batch_04.pkl','rb'))[0]['dataset']
tr=d['train']; labs=tr['lab_id']; n=len(labs)
dt,t,past,hold,x=s._prepare(d)
active,plat=s._activation(d,past,dt)
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']
args=past[:n],hold[:n],x[:n],active[:n],plat[:n]
missing=np.isnan(tr['coupon_features_raw']).any(axis=1)
for lab in (*np.unique(labs), -1):
    fit, test=labs!=lab, labs==lab
    if lab==-1: test=fit
    tau,w=s._fit(dt,t,*(a[fit] for a in args),yc[fit],yr[fit],~missing[fit])
    c,r=s._design(tau,dt,t,*(a[test] for a in args))
    ec,er=c@w-yc[test],r@w-yr[test]
    print('lab',lab,'tau',tau.round(5),'weights',w.round(5))
    for name,g in [('all',np.ones(len(ec),bool)),('complete',~missing[test]),('imputed',missing[test]),('damaged',active[:n][test]>0),('undamaged',active[:n][test]==0)]:
        if g.any(): print(name,g.sum(),'curve/rate RMSE',np.sqrt(np.mean(ec[g]**2)).round(5),np.sqrt(np.mean(er[g]**2)).round(5),'max',np.abs(ec[g]).max().round(5),np.abs(er[g]).max().round(5))
        if g.any(): print('median absolute, curve/rate',np.median(np.abs(ec[g])).round(5),np.median(np.abs(er[g])).round(5))
    if lab==-1:
        print('curve time RMSE',np.sqrt(np.mean(ec**2,axis=0)).round(5)); print('rate abs quantiles',np.quantile(np.abs(er),[.5,.9,.95,.99]))
print('act count train/test',plat[:n].sum(),plat[n:].sum())
