import numpy as np, pickle, runpy
from itertools import combinations
from glob import glob
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(glob('/tmp/hegel_continual_archive/*solver.py')[0]); n1=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; n2=runpy.run_path('/holdout_task/_frozen_nodes/node_02.py')['run_node']
u,v=(s['panel_a'][k] for k in ('u','v')); t=s['train']; opts=a['_panel_operators'](s['panel_c']); on=np.ones(len(u)); rng=np.random.default_rng(92); subs=[np.arange(len(u))]+[rng.choice(len(u),10,False) for _ in range(240)]
for mode in ['cross','cross_all_lin','cross_den','cross_den_both','cross_den_prod','cross_higher','fullquad']:
 def basis(x,i,k=0):
  y=x[:,i]; z=x[:,(i+1)%4]; yz=y*z; o=np.ones(len(x)); l=np.c_[o,y,z]; lall=np.c_[o,x]
  if mode=='cross': return np.c_[l,yz]
  if mode=='cross_all_lin': return np.c_[lall,yz]
  if mode.startswith('cross_den'): return np.c_[l,yz/(1+k*({'cross_den':z*z,'cross_den_both':y*y+z*z,'cross_den_prod':yz*yz}[mode]))]
  if mode=='cross_higher': return np.c_[l,yz,y*y*z,y*z*z,y*y*z*z]
  return np.c_[lall,x*x,*[x[:,j]*x[:,h] for j,h in combinations(range(4),2)]]
 ds=[]; model=[]
 for i in range(4):
  f=basis(u,i); target=v[:,i]; p=f.shape[1]; best=None
  for ix in subs if p<=10 else [subs[0]]+[rng.choice(len(u),p,False) for _ in range(260)]:
   for rep in range(5):
    w=np.linalg.lstsq(f[ix],target[ix],rcond=1e-9)[0]; d=np.abs(f@w-target); ix=np.argsort(d)[:37]
   j=np.mean(d[ix]**2)
   if best is None or j<best[0]: best=j,ix,w
  _,ix,w=best; k=0.
  for rep in range(4):
   if mode.startswith('cross_den'):
    def res(kk):
     h=basis(u,i,kk[0])[ix]; return h@np.linalg.lstsq(h,target[ix],rcond=None)[0]-target[ix]
    k=least_squares(res,[k],bounds=(0,5)).x[0]
   f=basis(u,i,k); w=np.linalg.lstsq(f[ix],target[ix],rcond=None)[0]; d=np.abs(f@w-target); med=np.median(d); ix=np.flatnonzero(d < max(1e-7,4*med));
  model.append((k,w)); ds.append(d); print(mode,i,round(k,4),np.round(w,4),np.round(np.sort(d)[[20,35,39,42]],5))
 def forward(x): return np.column_stack([basis(x,i,k)@w for i,(k,w) in enumerate(model)])
 d=np.linalg.norm(np.array(ds),axis=0); print('joint',np.round(np.sort(d)[[20,35,39,42,45]],5)); x=np.concatenate((t['x'],s['test']['x'])); x0=a['_calibration'](s['panel_a']); q0=np.linalg.solve(x0[1].T,(x-x0[0]).T).T
 inv=[]
 for y,q in zip(x,q0):
  r=least_squares(lambda z:forward(z[None,:])[0]-y,q,gtol=1e-12,max_nfev=80); inv.append(r.x)
 q=np.array(inv); print('inverse',np.max(np.linalg.norm(forward(q)-x,axis=1)),np.quantile(np.abs(q),[.9,.99,1])); common, off=a['_features'](*(f(q) for f in opts)); errs=[]
 for z in [np.arange(52,72),*[z for z in np.array_split(np.random.default_rng(889).permutation(72),4)]]:
  tr=np.setdiff1d(np.arange(72),z); sub={**s,'train':{k:t[k][tr] for k in ('x','y','traj')}}; temp=n1({'__input__':sub}); yh=a['_ridge'](common[tr],t['y'][tr],common[z],1e-6); out=n2({'node_00':{'common':common[np.r_[tr,z]],'offdiag':off[np.r_[tr,z]],'y_hat':yh},'node_01':temp}); errs.append([np.abs(yh-t['y'][z]).mean(),np.abs(out['state_pred']-t['traj'][z]).mean()])
 print('CV',mode,np.round(errs,6),flush=True)
