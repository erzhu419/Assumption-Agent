import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import *
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
ds = calibrate_upstream(dict(item['dataset'],chamber_verification_trials=item['campaign_diagnostics']))
tr,te,d = (ds[k] for k in ('train','test','chamber_verification_trials'))
dt,t=ds['past_dt'],ds['eval_times']
x,xt,xd=features(tr,te,d)
taus,theta=material(tr,x,dt,t)
print('dt,tau,theta',dt,taus,theta)
from batch_solver import solve_batch
from scipy.optimize import least_squares
xx=features(d,d)[0]; past,hold=loads(d); obs=d['strain_curves_obs']-d['strain_0'][:,None]; rate=d['instant_rate_obs']; missing=np.isnan(d['coupon_features_raw']).any(1)
def metrics(c,r):
    for label,m in [('all',np.ones(len(hold),bool)),('complete',~missing),('missing',missing)]+[(f'lab{l}',d['lab_id']==l) for l in np.unique(d['lab_id'])]:
        print(label, np.round([np.sqrt(np.mean((r[m]-rate[m])**2)),np.sqrt(np.mean((c[m]-obs[m])**2)),np.max(np.sqrt(np.mean((c[m]-obs[m])**2,axis=0)))],5))
print('reset: rate RMSE, strain RMSE, worst horizon strain RMSE')
metrics(*chamber(taus,theta,hold,xx,t))
raw=d['coupon_features_raw']; lab=d['lab_id']; group=d['stratum']; zz=raw.copy()
for l in np.unique(lab):
    m=lab==l; panel=d['ref_coupons'][m]; zz[m]=(zz[m]-np.median(panel.mean(0)))/np.median(panel.std(0))
_,cells,counts=np.unique(np.c_[lab,group],axis=0,return_inverse=True,return_counts=True)
Q=np.vstack((1/taus,-np.expm1(-t[:,None]/taus))); E=np.c_[rate,obs]-hold[:,None]*(Q@theta[:2]); B=np.c_[np.r_[1,t],Q*taus]; D=theta[2]+xx@theta[3:]; a=np.linalg.lstsq((D[~missing,None,None]*B).reshape(-1,3),E[~missing].ravel(),rcond=None)[0]; K=B@a; implied=E@K/(K@K)
for j in np.flatnonzero(np.isnan(zz).any(0)):
    valid=np.isfinite(zz[:,j]); m=~valid; present=np.bincount(cells[valid],minlength=len(counts));
    print('impute feature',j,'inferred',np.round(xx[m,j]+(implied-D)[m]/theta[3+j],4),'unweighted/inverse-count/inverse-valid-count',np.round([zz[valid,j].mean(),np.average(zz[valid,j],weights=1/counts[cells[valid]]),np.average(zz[valid,j],weights=1/present[cells[valid]])],4))
for k,v in solve_batch(item['dataset']).items():
    assert np.isfinite(v).all(); print(k,v.shape)
