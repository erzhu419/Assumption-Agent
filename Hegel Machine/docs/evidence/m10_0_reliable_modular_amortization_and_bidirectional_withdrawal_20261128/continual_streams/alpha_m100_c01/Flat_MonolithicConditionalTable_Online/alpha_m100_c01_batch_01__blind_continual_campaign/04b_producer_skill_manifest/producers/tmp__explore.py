import numpy as np,pickle,sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp'); import material as m
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr,te,di=ds['train'],ds['test'],item['campaign_diagnostics']; dt,t=ds['past_dt'],ds['eval_times']; cal=ds['calibration_load_cells']; x=cal['true_channel_loads']; y=cal['bridge_raw'];
def poly(x, degree):
 from itertools import combinations_with_replacement as comb
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for d in range(1,degree+1) for i in comb(range(4),d)])
def rms(x): return np.sqrt(np.mean(x*x))
def q(x):return np.round(np.quantile(abs(x),[.25,.5,.75,.9,.99,1]),6)
if __name__=='__main__':
 for a,b in [(x,y),(y,x)]:
  for d in (1,2,3):
   X=poly(a,d); n=len(X); fold=np.arange(n)%5; pred=np.empty_like(b)
   for f in np.unique(fold):
    keep=fold!=f; pred[~keep]=X[~keep]@np.linalg.lstsq(X[keep],b[keep],rcond=None)[0]
   print('cal', 'forward' if a is x else 'inverse',d,q(pred-b),q((pred-b).sum(1)))
bridge=m._bridge(y,x); ztr,zte=m._features(tr,te); _,zdi=m._features(dict((k,np.concatenate([tr[k],te[k]])) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']),di)
inputs=[m._inputs(s,z,bridge) for s,z in zip((tr,te,di),(ztr,zte,zdi))]; tau,theta=m._fit(*inputs[0],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs'])
if __name__=='__main__':
 print('tau theta bridge',tau,theta,bridge)
 for s,inp in [(tr,inputs[0]),(di,inputs[2])]:
  Y=s['strain_curves_obs']-s['strain_0'][:,None]; R=s['instant_rate_obs']; C,B=m._design(tau,*inp,dt,t); print('baseline residual',q(Y-C@theta),q(R-B@theta))
  T=np.column_stack([ -np.expm1(-t[:,None]/tau),t]); D=np.vstack([T,np.r_[1/tau,1]]); yy=np.c_[Y,R]; coeff=np.array([least_squares(lambda w:D@w-v,np.linalg.lstsq(D,v,rcond=None)[0],loss='soft_l1',f_scale=.008).x for v in yy]); p,h,F=inp; gap=B[:,:2]*tau; drift=F@theta[2:]; print('independent amp fit residual',q(yy-coeff@D.T)); print('baseline and fit, first 15',np.round(np.c_[gap*theta[:2],drift,coeff,(R-B@theta)],4)[:15]); print('ranges',np.round(np.c_[p,h].min(0),3),np.round(np.c_[p,h].max(0),3))
