import pickle,numpy as np
from scipy.optimize import minimize_scalar
from m100_base import _bridge,_loads,_features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=170)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=_features(tr,te); xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(q['lab_id']):]; b=_bridge(d['calibration_load_cells']); tau,theta=_fit(*_loads(tr,b),x[:n],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,h=_loads(q,b); C,R=_design(tau,p,h,xd,dt,t); ec=q['strain_curves_obs']-q['strain_0'][:,None]-C@theta; er=q['instant_rate_obs']-R@theta; E=np.c_[er,ec]
def temporal(k,rate=True):
 L=np.c_[-np.expm1(-t/k),t]; return np.vstack(([1/k,1],L)) if rate else L
def err(k,rate=True):
 L=temporal(k,rate); Y=E if rate else ec; return Y-(Y@np.linalg.pinv(L.T))@L.T
from scipy.optimize import least_squares
D=np.c_[q['instant_rate_obs'],q['strain_curves_obs']-q['strain_0'][:,None]]
keys=('lab_id','stratum','coupon_features_raw','ref_coupons','past_channel_raw','hold_channel_raw','strain_0')
for name,r in [('train',tr),('test',te)]:
 print('overlap',name,{k:int(np.any(np.all(np.isclose(np.asarray(q[k]).reshape(len(h),-1)[:,None,:],np.asarray(r[k]).reshape(len(r['lab_id']),-1)[None,:,:],atol=1e-8,rtol=0,equal_nan=True),axis=2),axis=1).sum()) for k in keys})
raw=q['coupon_features_raw']; lab=q['lab_id']; O=np.eye(4)[lab]; valid=~np.isnan(raw).any(1); T=np.r_[1.,t]; L=lambda k:np.r_[1/k,-np.expm1(-t/k)]
F=np.c_[h,O,O*(np.nan_to_num(raw)@theta[3:])[:,None]]; B=np.linalg.lstsq(F[valid],D[valid],rcond=None)[0]; print('complete affine time profiles; hold, mean intercept, slopes normalized')
print(np.vstack([B[0],B[1:5].mean(0),B[5:].mean(0)]))
for fitlinear in (False,True):
 def mat(k): return np.array([L(kk) for kk in k]+([T] if fitlinear else []))
 def loss(k):
  K=mat(k); return ((F[valid]@B@np.linalg.pinv(K)@K)-F[valid]@B).ravel()
 opt=least_squares(loss,[.58,4.2],bounds=([.25,1.],[1.,20.])); C=B@np.linalg.pinv(mat(opt.x)); print('fitlinear,taus,rmse,coef',fitlinear,opt.x,np.mean(loss(opt.x)**2)**.5,C)
for labid in range(4):
 mask=lab==labid; print('q feature nan rows and refs',labid,np.where(mask&~valid)[0],q['ref_coupons'][mask][:2])
