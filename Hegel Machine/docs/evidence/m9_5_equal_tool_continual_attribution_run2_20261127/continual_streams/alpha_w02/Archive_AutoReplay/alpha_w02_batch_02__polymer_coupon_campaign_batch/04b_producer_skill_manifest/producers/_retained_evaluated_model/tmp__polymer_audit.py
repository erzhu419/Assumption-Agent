import importlib.util,pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w02_batch_02_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_alpha_w02_batch_02.pkl','rb'))[0]; d=p['dataset']; s=d['train']; q=d['test']; t=d['eval_times']; dt=d['past_dt']; l=s['lab_id']; n=len(l); X=m._features(s,q)[:n]; miss=~np.isfinite(s['coupon_features_raw']).all(1); st=m._stress(s,m._calibrate(d['calibration_load_cells'],d['yield_stress'])); y=np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]; den=np.maximum(abs(st[1]),.5)
print('training fields, test plastic states and largest absolute training stresses',list(s),np.unique(q['residual_plastic_state']),max(np.max(abs(st[0])),np.max(abs(st[1]))),d['yield_stress']); print('stratum counts by lab',[(int(i),np.bincount(s['stratum'][l==i]).tolist()) for i in np.unique(l)])
# Every challenger is fixed before scoring; train labels never enter its held-out lab fit.
results={k:[] for k in ['base','pairs','quotient','stress','memory_x','stratum']}
for i in np.unique(l):
 a=(l!=i)&~miss; b=(l==i)&~miss
 tau,p=m._fit(tuple(x[a] for x in st),X[a],dt,t,y[a]); A=m._design(st,X,dt,t,np.log(tau)); pred=A@p; r=y-pred
 x=X[:,1:]; u=st[1]/d['yield_stress']; v=m._memory(st,dt,tau); D=np.r_[t,1.];
 extras={'pairs':np.c_[x*x,*[x[:,j]*x[:,k] for j in range(4) for k in range(j)]], 'quotient':np.c_[x/(1+u[:,None]**2)-x,u/(1+u*u)],'stress':np.c_[u,u*u,u**3,x*u[:,None]],'stratum':np.eye(3)[s['stratum']][:,1:]}
 extras={k:z[:,None,:]*D[None,:,None] for k,z in extras.items()}; extras['memory_x']=(A[:,:,:2,None]*x[:,None,None,:]).reshape(len(l),len(D),-1)
 for k in results:
  if k=='base': e=r[b,:-1]
  else:
   B=extras[k]; W=B[a].reshape(-1,B.shape[-1]); target=r[a].ravel(); c=np.linalg.lstsq(W,target,rcond=None)[0]
   for _ in range(12):
    w=1/np.hypot(1,(W@c-target)/.03);c=np.linalg.lstsq(W*w[:,None],target*w,rcond=None)[0]
   e=r[b,:-1]-B[b,:-1]@c
  results[k].append([np.median(abs(e)),np.median(abs(e[:,-1]/den[b])),np.median(abs(e@t/(t@t)))])
 print('lab',i,'tau',np.round(tau,5),'complete baseline median rate / projected drift residual',np.round([np.median(r[b,-1]),np.median(r[b,:-1]@t/(t@t))],5))
for k,z in results.items(): print(k,'worst lab median absolute strain/secant/drift',np.round(np.max(z,0),6),'mean of labs',np.round(np.mean(z,0),6))
