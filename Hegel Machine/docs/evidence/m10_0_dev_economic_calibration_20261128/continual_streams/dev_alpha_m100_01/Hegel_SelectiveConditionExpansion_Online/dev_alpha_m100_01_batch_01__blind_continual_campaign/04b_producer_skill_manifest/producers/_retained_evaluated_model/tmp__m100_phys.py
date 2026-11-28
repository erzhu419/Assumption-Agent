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
raw=q['coupon_features_raw']; lab=q['lab_id']; O=np.eye(4)[lab]; valid=~np.isnan(raw).any(1); T=np.r_[1.,t]; L=lambda k:np.r_[1/k,-np.expm1(-t/k)]
U=np.nan_to_num(raw)@theta[3:]; F=np.c_[h,O,O*U[:,None]]; M=valid
for hist in (0,1):
 V=np.c_[F,p] if hist else F
 B=np.linalg.lstsq(V[M],D[M],rcond=None)[0]; print('hist',hist,'linear projected rms',np.mean((V@B-D)[M]**2)**.5)
 for mode in ('fixed','fixedfast','free','free+linear','free+linearfeature'):
  def calc(k, ret=False):
   K=np.array([L(j) for j in k[:2]]+([T] if 'linear' in mode else [])); Z=(V[M,:,None,None]*K.T[None,None,:,:]).transpose(0,2,1,3).reshape(M.sum()*len(T),-1); keep=np.ones((V.shape[1],len(K)),bool)
   if mode=='free+linearfeature': keep[0,-1]=False
   Z=Z[:,keep.ravel()]; coeff=np.linalg.lstsq(Z,D[M].ravel(),rcond=None)[0]; res=Z@coeff-D[M].ravel(); c=np.zeros_like(keep,dtype=float); c[keep]=coeff
   return (c,res) if ret else res
  k=tau if mode=='fixed' else np.r_[tau[0],least_squares(lambda s:calc(np.r_[tau[0],s]),[4.2],bounds=(1,15)).x] if mode=='fixedfast' else least_squares(calc,[.58,4.2],bounds=([.3,1.],[1.,15.])).x
  c,e=calc(k,True); print(mode,k,'rate,curve',np.sqrt(np.mean(e.reshape(-1,len(T))**2,axis=0))[[0,-1]],'coefs',[c[0].round(4),c[1:5].mean(0).round(4),c[5:9].mean(0).round(4)],'history',c[9:].round(4).tolist())
print('panel scales');
for P in [(tr,q),(tr,te,q),(tr,te),(q,), (te,q)]:
 labp=np.concatenate([a['lab_id'] for a in P]); refs=np.concatenate([a['ref_coupons'] for a in P]); print([np.median(refs[labp==j].std(0)) for j in range(4)])
print('train, diag, test ranges hold history',[(np.min(_loads(a,b)[1]),np.max(_loads(a,b)[1]),np.min(_loads(a,b)[0]),np.max(_loads(a,b)[0])) for a in (tr,q,te)])
