import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/alpha_m97_c03_batch_03_solver.py'; sp=importlib.util.spec_from_file_location('s',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))[0]['dataset']; a,b=d['train'],d['test']
x=m._align(a,b)[:len(a['lab_id'])]; past,hold=m._physical(a,m._calibrate(d['calibration_load_cells'])); dt,t=d['past_dt'],d['eval_times']; y=np.column_stack((a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']))
def robust(keep, sc):
 args=(past[keep],hold[keep],dt,t,x[keep]); obs=y[keep]; lo=np.log([.015,.04]); hi=np.log([20.,200.]); best=None
 for f,s in ((.25,1.6),(.55,3.2),(1.2,7.)):
  p=np.log([f,s/f-1]); w=np.linalg.lstsq(m._design(np.array([f,s]),*args).reshape(-1,9),obs.ravel(),rcond=None)[0]
  def residual(v):
   f,r=np.exp(v[:2]); return ((m._design(np.array([f,f*(1+r)]),*args)@v[2:]-obs)/sc).ravel()
  z=np.r_[p,w]
  for loss,width in [('soft_l1',3),('cauchy',1)]:
   sol=least_squares(residual,z,loss=loss,f_scale=width,bounds=(np.r_[lo,np.full(9,-np.inf)],np.r_[hi,np.full(9,np.inf)]),max_nfev=180,x_scale='jac'); z=sol.x
  if best is None or sol.cost < best.cost: best=sol
 f,r=np.exp(best.x[:2]); return np.array([f,f*(1+r)]),best.x[2:]
for method in ('ols',.003,.015,.05):
 all_e=np.empty_like(y)
 for c in [-1,0,1,2,3]:
  keep=a['lab_id']!=c; args=(past[keep],hold[keep],dt,t,x[keep],y[keep]); tau,w=m._fit(*args) if method=='ols' else robust(keep,method)
  sel=~keep if c>=0 else keep; e=m._design(tau,past[sel],hold[sel],dt,t,x[sel])@w-y[sel]
  print('fit',method,c,'tau',np.round(tau,6),'med/p95',np.round(np.quantile(abs(e[:,:-1]),[.50,.95]),6),np.round(np.quantile(abs(e[:,-1]),[.50,.95]),6),'rate rmse',round(np.sqrt(np.mean(e[:,-1]**2)),6))
  if c<0: print('w',np.round(w,6),'counts > .1/1',[(abs(e[:,:-1])>v).sum() for v in [.1,1]],'by horizon scaled MAD',np.round(1.4826*np.median(abs(e-np.median(e,axis=0)),axis=0),6))
  else: all_e[sel]=e
 for k,sel in [('missing',np.isnan(a['coupon_features_raw']).any(axis=1)),('complete',~np.isnan(a['coupon_features_raw']).any(axis=1))]: print(method,k,'med abs',np.median(abs(all_e[sel,:-1])),np.median(abs(all_e[sel,-1])))
