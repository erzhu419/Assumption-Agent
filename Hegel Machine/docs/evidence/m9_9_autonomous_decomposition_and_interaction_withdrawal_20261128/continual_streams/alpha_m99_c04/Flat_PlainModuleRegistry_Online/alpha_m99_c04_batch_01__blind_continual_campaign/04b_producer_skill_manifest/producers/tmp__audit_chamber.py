import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
d,g=item['dataset'],item['campaign_diagnostics']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; s=m._stress_map(d['calibration_load_cells'])
xt,_=m._features(tr,te); x=m._chamber_features(g); out=m.solve_batch(d); u,b=out['estimated_taus'],out['hereditary_weights']
h=s[0]+g['hold_channel_raw']@s[1:]; p=s[0]+g['past_channel_raw']@s[1:]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]
K=np.vstack([-np.expm1(-t[:,None]/u),1/u]); L=np.r_[np.log1p(t),1.]; hist=p@(-np.expm1(-dt[:,None]/u)*np.exp(-age[:,None]/u)); r=b[2]+x@b[3:]
Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; base=(h[:,None]*b[:2])@K.T+np.outer(r,L); E=Y-base; labs=g['lab_id']
rmse=lambda e: np.sqrt(np.mean(e**2)); score=lambda e: [rmse(e[:,:-1]),rmse(e[:,-1])]
miss=np.isnan(g['coupon_features_raw']).any(1)
print('fixed law worst lab, stratum, and missingness componentwise RMSE:',*[np.round(np.max([score(E[v==j]) for j in np.unique(v)],axis=0),6) for v in (labs,g['stratum'],miss)])
corrections={'history':np.stack([np.outer(hist[:,j],K[:,j]) for j in range(2)],axis=-1),'pairwise':np.stack([np.outer((x[:,i]+sign*x[:,j])**2,L) for i in range(4) for j in range(i+1,4) for sign in (-1,1)],axis=-1),'lab offset':np.outer(np.ones(len(x)),L)[:,:,None]}
for name,A in corrections.items():
 errs=[]
 for lab in np.unique(labs):
  i=labs!=lab; c=np.linalg.lstsq(A[i].reshape(-1,A.shape[-1]),E[i].ravel(),rcond=None)[0]; errs.append(score(E[~i]-A[~i]@c))
 print(name,'worst held-out lab strain/rate',np.round(np.max(errs,axis=0),6))
errs=[]
for lab in np.unique(labs):
 i=labs!=lab
 def corr(a): return np.outer(r*(a[0]/(1+a[1]*r*r)-1),L)
 a=least_squares(lambda a:(E[i]-corr(a)[i]).ravel(),[1,0],bounds=([0,0],[2,10])).x
 errs.append(score(E[~i]-corr(a)[~i]))
print('rational attenuation worst held-out lab strain/rate',np.round(np.max(errs,axis=0),6))
base_errors=[]; params=[]
for lab in np.unique(tr['lab_id']):
 i=tr['lab_id']!=lab
 split=lambda mask:{k:v[mask] for k,v in tr.items()}
 ub,bb=m._fit(split(i),xt[i],s,dt,t); c,q=m._design(split(~i),xt[~i],s,dt,t,ub)
 e=c@bb+tr['strain_0'][~i,None]-tr['strain_curves_obs'][~i]; z=q@bb-tr['instant_rate_obs'][~i]; base_errors.append([rmse(e),rmse(z),np.median(abs(e)),np.median(abs(z))]); params.append(np.r_[ub,bb])
print('train leave-one-lab-out worst strain/rate RMSE and median absolute errors',np.round(np.max(base_errors,axis=0),6))
print('maximum parameter changes under lab withdrawal',np.round(np.max(np.abs(np.array(params)-np.r_[u,b]),axis=0),6))
