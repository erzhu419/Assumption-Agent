import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_m99_c01_s00_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];tr=d['train'];te=d['test'];c=i['campaign_diagnostics'];dt=d['past_dt'];t=d['eval_times'];cal=d['calibration_load_cells'];raw=cal['bridge_raw'];y=cal['true_channel_loads'];b=m._bridge(cal)
print('bridge',b,'raw range',np.min(raw,0),np.max(raw,0)); print('sweeps',[(k,np.shape(v),np.asarray(v).ravel()[:8]) for k,v in cal['single_channel_sweeps'].items()])
from itertools import combinations_with_replacement
for degree in (1,2,3):
 A=np.column_stack([np.ones(len(raw))]+[np.prod(raw[:,j],axis=1) for j in range(1,degree+1) for j in combinations_with_replacement(range(4),j)])
 pred=np.zeros_like(y)
 for q in np.array_split(np.random.default_rng(5).permutation(len(y)),5):
  z=np.setdiff1d(np.arange(len(y)),q);pred[q]=A[q]@np.linalg.lstsq(A[z],y[z],rcond=None)[0]
 e=pred-y;print('bridge OOF degree',degree,'channel rmse',np.sqrt(np.mean(e**2,0)),'stress rmse',np.sqrt(np.mean(e.sum(1)**2)))
x,xt=m._features(tr,te);p,h=m._loads(tr,b);pc,hc=m._loads(c,b);tau,beta=m._fit(x,p,h,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,beta)
_,xc=m._features(dict((k,np.concatenate([tr[k],te[k]],axis=0)) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')),c)
X=m._features(c,c)[1]; D=beta[2]+X@beta[3:]; H=hc; P=pc; y=np.c_[c['strain_curves_obs']-c['strain_0'][:,None],c['instant_rate_obs']]; complete=~np.isnan(c['coupon_features_raw']).any(1)
lab=c['lab_id']; st=c['stratum']; age=np.cumsum(dt[::-1])[::-1]-dt; z=P@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau))
def errors(e):return np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))
def project(kind, use, ta):
 F=np.vstack((np.column_stack((-np.expm1(-t[:,None]/ta),t)),np.r_[1/ta,1.])); base=(H[:,None]*beta[:2])@F[:,:2].T; cols=[D[:,None]*F[None,:,j] for j in (0,1)];
 if kind=='retain': cols+=[z[:,j,None]*F[None,:,j] for j in (0,1)]
 if kind=='drift': cols+=[D[:,None]*F[None,:,2]]
 if kind=='offset': cols+=[np.broadcast_to(F[:,j],y.shape) for j in (0,1,2)]
 A=np.stack(cols,axis=-1); AA=A[use].reshape(-1,A.shape[-1]); yy=(y-base)[use].ravel(); coef=np.linalg.lstsq(AA,yy,rcond=None)[0]; return A@coef+base-y,coef
folds=[lab==k for k in np.unique(lab)]+[st==k for k in np.unique(st)]
for kind in ('reset','retain','drift','offset'):
 for free in (False,True):
  def fit(use):
   ta=np.exp(least_squares(lambda u:project(kind,use,np.exp(u))[0][use].ravel(),np.log(tau),bounds=(np.log([.58,3]),np.log([.83,7]))).x) if free else tau
   return (*project(kind,use,ta),ta)
  e,q,ta=fit(complete); out=np.max([errors(fit(complete&~f)[0][f&complete]) for f in folds if np.any(f&complete)],axis=0); print(kind,free,np.round(ta,5),np.round(q,5),'complete/missing',np.round([errors(e[complete]),errors(e[~complete])],5),'worst complete',out.round(5))
