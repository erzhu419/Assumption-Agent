import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _fit,_features,_design
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
raw=item['dataset'];print('dataset keys:',list(raw))
data=calibrate_upstream(dict(raw,chamber_verification_trials=item['campaign_diagnostics']))
tr,te,q=(data[k] for k in ['train','test','chamber_verification_trials']); dt=data['past_dt'];t=data['eval_times']
x,_=_features(tr,te); x=np.c_[np.ones(len(x)),x]; p=tr['past_channel_raw'].sum(-1);h=tr['hold_channel_raw'].sum(-1)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; obs=np.c_[y,r];labs=tr['lab_id']

def residual(a):
 c,rr=_design(a[:2],p,h,x,dt,t)
 return np.c_[c@a[2:],rr@a[2:]]-obs

def refits(rows):
 a,b=_fit(p[rows],h[rows],x[rows],dt,t,y[rows],r[rows]); init=np.r_[a,b]; e=residual(init); keep=(abs(e[rows])<.06);print('excluded strain/rate',np.sum(~keep[:,:-1]),np.sum(~keep[:,-1]))
 yield init
 for w in (1.,2.):
  weights=np.r_[np.ones(len(t)),w]
  yield least_squares(lambda v:(residual(v)[rows]*weights)[keep],init,x_scale='jac').x

names=('base','trim','trim_rate2');errs=np.empty((3,*obs.shape))
for l in np.unique(labs):
 rows=labs==l
 for j,a in enumerate(refits(~rows)):
  errs[j,rows]=residual(a)[rows]
for name,e in zip(names,errs):
 print(name,'train worst lab strain/rate P90',max(np.quantile(abs(e[labs==l,:-1]),.9) for l in np.unique(labs)),max(np.quantile(abs(e[labs==l,-1]),.9) for l in np.unique(labs)))

_,xq=_features(q,q);hq=q['hold_channel_raw'].sum(-1);yq=q['strain_curves_obs']-q['strain_0'][:,None];rq=q['instant_rate_obs'];lq=q['lab_id'];complete=np.isfinite(q['coupon_features_raw']).all(1)
for name,a in zip(names,refits(np.ones(len(labs),bool))):
 ts=a[:2]; b=a[2:];dr=b[2]+xq@b[3:];F=-np.expm1(-t[:,None]/ts); Y=yq-hq[:,None]*(F@b[:2])-dr[:,None]*t; M=dr[:,None,None]*(F*ts-t[:,None]); R=rq-hq*(b[:2]@(1/ts))-dr; E=np.empty_like(Y)
 for l in np.unique(lq):
  rows=(lq!=l)&complete
  coeff=np.linalg.lstsq(M[rows].reshape(-1,2),Y[rows].ravel(),rcond=None)[0]; E[lq==l]=Y[lq==l]-M[lq==l]@coeff
 rows=complete; coeff=np.linalg.lstsq(M[rows].reshape(-1,2),Y[rows].ravel(),rcond=None)[0]
 print(name,'full parameters',np.round(a,7),'kernel',np.round(coeff,7),'diagnostic CV complete strain/rate RMSE',np.sqrt(np.mean(E[rows]**2)),np.sqrt(np.mean(R[rows]**2)),'worst lab strain/rate RMSE',max(np.sqrt(np.mean(E[lq==l]**2)) for l in np.unique(lq)),max(np.sqrt(np.mean(R[lq==l]**2)) for l in np.unique(lq)))
