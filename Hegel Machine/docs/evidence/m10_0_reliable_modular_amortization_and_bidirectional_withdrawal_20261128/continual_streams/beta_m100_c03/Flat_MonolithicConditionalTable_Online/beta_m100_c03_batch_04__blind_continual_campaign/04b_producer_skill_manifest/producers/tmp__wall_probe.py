import sys,pickle,itertools
import numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import beta_m100_c03_batch_03_solver as base
import beta_m100_c03_batch_01_solver as response
import beta_m100_c03_batch_02_solver as periodic
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; raw=s['y_obs_raw']; y=base._calibrate(s,raw); z=s['Z_nuisance']; lam=s['nuisance_ridge']; ridge=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; sensor=s['sensor_ij']; ns=len(sensor)
print('calibrated range',y.min(),y.max(),'shifts',s['mu_shifts'])
def axes(modes):
 out=[]
 for d,mode in zip('xy',modes):
  k,v=s['kappa_'+d],s['vel_'+d]; h=s['L'+d]/(s['n'+d]-1); rob=s['robin_'+d]
  out.append(periodic._axis(k,v,h) if mode=='periodic' else base._axis(k,v,h,rob if mode=='robin' else 0.0))
 return out
def fit(S,ix,reg):
 X=base._profile(dict(s,Z_nuisance=z[ix]),np.column_stack((S[ix],y[ix]))); F,b=X[:,:-1],X[:,-1]; w,_=base._box_qp(F.T@F+reg*np.eye(S.shape[1]),-F.T@b,cap,budget); gam=np.linalg.solve(z[ix].T@z[ix]+lam*np.eye(z.shape[1]),z[ix].T@(y[ix]-S[ix]@w)); return w,gam
folds=[[i+k*ns for k in range(len(s['mu_shifts']))] for i in range(ns)]
for modes in itertools.product(('robin','sealed','periodic'),repeat=2):
 S,_=response._responses(s,axes(modes)); scores=[]
 for reg in (1e-8,ridge):
  pred=np.zeros_like(y)
  for te in folds:
   tr=np.setdiff1d(np.arange(len(y)),te); w,g=fit(S,tr,reg); pred[te]=S[te]@w+z[te]@g
  w,g=fit(S,np.arange(len(y)),reg); err=y-pred; scores.append((round(np.sqrt(np.mean((y-S@w-z@g)**2)),7),round(np.sqrt(np.mean(err**2)),7),np.round(np.sqrt(np.mean(err.reshape(-1,ns)**2,axis=1)),7).tolist(),round(max(np.sqrt(np.mean(err[te]**2)) for te in folds),7)))
 print(modes,scores)
