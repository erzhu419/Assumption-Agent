import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep01_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
for case in pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb')):
 d=case['dataset']; p=d['train']; t=d['eval_times']; dt=d['past_dt']; lab=p['lab_id']; n=len(lab)
 x, complete=s._align(d); x=x[:n]; complete=complete[:n]
 g,b=s._calibrate(d['calibration_load_cells']); all_d=s._Design(p,x,g,b,dt,t)
 y=np.c_[p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']]
 stats={k:[] for k in ('base','anchored_fit','full_rate','precision_weight')}
 def predict(q, weight):
  v=all_d.matrix(q[:2])@q[2:]; return v[:,:-1]+(weight*(y[:,-1]-v[:,-1]))[:,None]*t
 for l in np.unique(lab):
  fit=complete & (lab!=l); miss=(~complete)&(lab!=l); val=lab==l
  fd=s._Design(p,x,g,b,dt,t,fit); md=s._Design(p,x,g,b,dt,t,miss)
  a,c=s._fit(fd,y[fit]); q=np.r_[a,c]
  scale=s._noise(fd.matrix(a)@c-y[fit]); ms=np.hypot(scale[0],t*scale[-1]); target=y[miss,:-1]-t*y[miss,-1,None]
  def resid(v):
   f=(fd.matrix(v[:2])@v[2:]-y[fit])/scale
   m=md.matrix(v[:2])[:,:,:2]@v[2:4]
   return np.r_[f.ravel(), ((m[:,:-1]-t*m[:,-1,None]-target)/ms).ravel()]
  q2=least_squares(resid,q,loss='cauchy',f_scale=2.5,x_scale='jac',max_nfev=100).x
  # Use the fitted linear drift's missing-feature variance and an initial-rate noise floor.
  present=np.isfinite(p['coupon_features_raw']); dev=x[fit].std(axis=0)*c[3:]; var=(~present)@(dev**2)
  weight=(var+.2*scale[-1]**2)/(var+scale[-1]**2)
  predictions={'base':predict(q,np.where(complete,.20,.98)), 'anchored_fit':predict(q2,np.where(complete,.20,.98)), 'full_rate':predict(q,np.where(complete,.20,1)), 'precision_weight':predict(q,weight)}
  for key,pred in predictions.items():
   for rows in (val&complete,val&~complete):
    if not rows.any(): continue
    err=abs(pred[rows]-y[rows,:-1]); den=np.maximum(abs(all_d.hold[rows]),.5)
    stats[key].append([np.minimum(err,.1).mean(),np.median(err[:,-1]/den),np.median(err[:,-1])])
  print('lab',int(l),'missing_n',int((val&~complete).sum()),'tau_baseline_vs_anchored',np.exp(q[:2]).round(5),np.exp(q2[:2]).round(5),'missing_precision_weight',np.quantile(weight[val&~complete],[0,.5,1]).round(4))
 print('worst lab/completeness [capped strain MAE, median secant AE, median endpoint AE]',{k:np.max(v,axis=0).round(6).tolist() for k,v in stats.items()})
