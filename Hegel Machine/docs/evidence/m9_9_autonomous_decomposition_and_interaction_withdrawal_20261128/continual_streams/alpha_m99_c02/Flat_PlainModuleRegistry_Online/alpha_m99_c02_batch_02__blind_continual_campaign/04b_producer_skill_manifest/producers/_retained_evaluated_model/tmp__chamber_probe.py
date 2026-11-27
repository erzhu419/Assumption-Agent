import sys,pickle,json,importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_m99_c02_batch_01_solver.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['dataset']; dt,t=raw['past_dt'],raw['eval_times']; ds=calibrate_upstream(raw); tr,te=ds['train'],ds['test']; di=calibrate_upstream({**raw,'test':item['campaign_diagnostics']})['test']; bridge=np.r_[0.,np.ones(4)]; xt,xe=old._features(tr,te); _,xd=old._features(tr,di); taus,theta=old._material(tr,bridge,xt,dt,t); print('taus theta',taus,theta)
def rms(a,axis=None):return np.sqrt(np.mean(a*a,axis=axis))
def report(s,x):
 p,h=old._loads(s,bridge); c,r=old._basis(taus,p,h,x,dt,t); y=s['strain_curves_obs']-s['strain_0'][:,None]; q=s['instant_rate_obs']; e=y-c@theta; er=q-r@theta
 print('median abs errors by time',np.round(np.median(abs(e),0),6),'rate',np.median(abs(er)),'RMS',rms(e),rms(er)); print('median signed',np.round(np.median(e,0),6),np.median(er)); print('rms by coupon',np.round(rms(e,1),5)); print('baseline curves',np.round((c@theta)[:3],5)); print('actual curves',np.round(y[:3],5)); print('hold & past range', h.min(),h.max(),p.min(),p.max());
 return p,h,c,r,y,q
report(tr,xt); p,h,c,r,y,q=report(di,xd)
from scipy.optimize import least_squares
for free in (False,True):
 def fun(a,ret=False):
  TT=np.exp(a) if free else taus; B=np.column_stack((-np.expm1(-t[:,None]/TT),t)); br=np.r_[1/TT,1.]; B=np.vstack((B,0.25*br)); Y=np.c_[y,.25*q]; co=Y@np.linalg.pinv(B.T); err=co@B.T-Y
  return (TT,co,err) if ret else err.ravel()
 a=least_squares(fun,np.log(taus),loss='soft_l1',f_scale=.001).x if free else []
 TT,co,err=fun(a,True); print('free',free,'effective taus',TT,'projected errors',np.median(abs(err),0)); print('amplitude ratios (quantile)',np.quantile(co[:,:2]/(r[:,:2]*taus*theta[:2]),[.1,.5,.9],axis=0),'drift difference',np.quantile(co[:,2]-r[:,2:]@theta[2:],[0,.1,.5,.9,1])); print('amp delta corr [hold,past,coupon]',np.round(np.corrcoef(np.c_[co[:,:2]-r[:,:2]*taus*theta[:2],co[:,2]-r[:,2:]@theta[2:],h,p,xd].T)[:3,3:],2))
