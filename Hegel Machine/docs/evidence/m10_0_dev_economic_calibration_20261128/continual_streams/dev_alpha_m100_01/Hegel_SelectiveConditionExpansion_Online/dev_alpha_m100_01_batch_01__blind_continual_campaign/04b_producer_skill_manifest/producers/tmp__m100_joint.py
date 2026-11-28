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
L=lambda k:np.r_[1/k,-np.expm1(-t/k)]
T=np.r_[1.,t]
parts=('lab_id','stratum','coupon_features_raw','ref_coupons')
allp={k:np.r_[tr[k],te[k]] for k in parts}; labs=allp['lab_id']; refs=allp['ref_coupons']; xx=q['coupon_features_raw'].copy()
for lab in np.unique(labs):
 panel=refs[labs==lab]; m=q['lab_id']==lab; xx[m]=(xx[m]-np.median(panel.mean(0)))/np.median(panel.std(0))
cell,idx,ct=np.unique(np.c_[labs,allp['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/ct[idx]; mu=np.nansum(x*w[:,None],axis=0); print('pooled imputed means',np.average(x,weights=w,axis=0)); xx=np.where(np.isnan(xx),np.average(x,weights=w,axis=0),xx)
for label, X in [('pooled',xd),('train-test frame',xx),('train+q',_features(tr,q)[-len(h):]),('test+q',_features(te,q)[-len(h):])]:
 print('\nFRAME',label)
 G=np.c_[np.ones(len(h)),h,p,X]; GS=np.c_[np.ones(len(h)),h,p,X@theta[3:]]
 for model in ('same taus','free slow','free both','no linear','shared feature','intrinsic+third'):
  F=GS if model=='shared feature' else G
  def temporal(ks):
   if model=='same taus': return np.array([L(tau[0]),L(tau[1]),T])
   if model=='free slow': return np.array([L(tau[0]),L(ks[0]),T])
   if model=='intrinsic+third': return np.array([L(tau[0]),L(tau[1]),L(ks[0]),T])
   if model=='no linear': return np.array([L(ks[0]),L(ks[1])])
   return np.array([L(ks[0]),L(ks[1]),T])
  def calc(ks,m,ret=False):
   K=temporal(ks); B=np.linalg.pinv(F[m])@D[m]@np.linalg.pinv(K); P=F@B@K
   return (P,B) if ret else (P[m]-D[m]).ravel()
  def fit(m):
   if model=='same taus': return tau
   k=[3.] if model in ('free slow','intrinsic+third') else [.58,3.]; bounds=([.08],[15.]) if len(k)==1 else ([.2,1.1],[1.05,15.]); return least_squares(calc,k,args=(m,),bounds=bounds,max_nfev=65).x
  m=np.ones(len(h),bool); k=fit(m); P,B=calc(k,m,True); err=P-D; folds=[]
  for lab in np.unique(q['lab_id']):
   m=q['lab_id']!=lab; cv,_=calc(fit(m),m,True); e=(cv-D)[~m]; folds.append(np.sqrt(np.mean(e**2,axis=0)))
  print(model,k,'train rate/strain',np.sqrt(np.mean(err[:,0]**2)),np.sqrt(np.mean(err[:,1:]**2)),'LOLO max rate/time',np.max(folds,axis=0)[0],np.max(np.array(folds)[:,1:]))
  if label in ('pooled','train-test frame') and model in ('same taus','free both','no linear','shared feature','intrinsic+third'): print('coefs: constant,hold,history,features\n',B.round(5))
