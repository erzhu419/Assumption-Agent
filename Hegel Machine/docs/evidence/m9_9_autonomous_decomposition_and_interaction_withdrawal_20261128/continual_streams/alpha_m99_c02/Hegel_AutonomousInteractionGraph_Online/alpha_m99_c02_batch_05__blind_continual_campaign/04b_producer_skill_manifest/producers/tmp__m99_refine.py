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
from scipy.optimize import least_squares
from itertools import product
raw=d['coupon_features_raw']; labs=d['lab_id']; xx=raw.copy()
for l in np.unique(labs):
    m=labs==l; p=d['ref_coupons'][m]; xx[m]=(xx[m]-np.median(p.mean(0)))/np.median(p.std(0))
xx=np.where(np.isnan(xx),np.nanmean(xx,axis=0),xx)
D=theta[2]+xx@theta[3:]; Q=np.vstack((1/taus,-np.expm1(-t[:,None]/taus))); V=np.r_[1,t]; Y=np.c_[rate,obs]; E=Y-hold[:,None]*(Q@theta[:2]); B=np.c_[V,Q*taus]
def kernel(mode,mask):
    y=E[mask]; u=D[mask,None]
    if mode=='nominal': return B@np.array([.25,.25,.5]), []
    if mode=='constrained_mix':
        C=B[:,1:]-B[:,:1]; w=np.linalg.lstsq((u[:,:,None]*C).reshape(-1,2),(y-u*V).ravel(),rcond=None)[0]; return V+C@w, np.r_[1-w.sum(),w]
    if mode=='free_mix':
        w=np.linalg.lstsq((u[:,:,None]*B).reshape(-1,3),y.ravel(),rcond=None)[0]; return B@w,w
    if mode=='one_tau':
        f=lambda a:.25*V+.25*B[:,1]+.5*np.r_[1,-np.expm1(-t/a[0])*a[0]]
        p=least_squares(lambda a:(y-u*f(a)).ravel(),[taus[1]],bounds=(.1,20)).x; return f(p),p
    if mode=='two_taus':
        f=lambda a:.25*V+np.vstack((1/a,-np.expm1(-t[:,None]/a)))@(a*np.array([.25,.50]))
        p=least_squares(lambda a:(y-u*f(a)).ravel(),taus,bounds=(.1,20)).x; return f(p),p
    if mode=='fixed_025_025_050_round':
        a=np.round(taus,1); return .25*V+np.vstack((1/a,-np.expm1(-t[:,None]/a)))@(a*np.array([.25,.50])),a
for mode in ('nominal','constrained_mix','free_mix','one_tau','two_taus','fixed_025_025_050_round'):
    k,p=kernel(mode,~missing); pred=np.zeros_like(Y)
    for l in np.unique(labs):
        j=labs==l; kl,_=kernel(mode,~(j|missing)); pred[j]=D[j,None]*kl
    err=pred-E; print(mode,'params',np.round(p,6),'all',np.round(np.sqrt(np.mean(err**2,0))[[0,-1]],5)); print('worst rate/strain/horizon',np.round(np.max([np.r_[np.sqrt(np.mean(err[m,0]**2)),np.sqrt(np.mean(err[m,1:]**2)),np.sqrt(np.mean(err[m,1:]**2,axis=0)).max()] for m in [labs==l for l in np.unique(labs)]+[missing,~missing]],axis=0),5))
