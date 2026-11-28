import os,sys,pickle,importlib.util,itertools as it,numpy as np
os.chdir('/holdout_task'); s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; tr=s['train'];
sp=importlib.util.spec_from_file_location('f','_frozen_nodes/node_00.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f); o=f.run_node({'__input__':s});print('frozen',[(k,np.shape(v)) for k,v in o.items()],'telemetry',s['telemetry_vec'])
def aff(u):return np.c_[np.ones(len(u)),u]
def poly(u,d):return np.column_stack([np.prod(u[:,idx],axis=1) for j in range(d+1) for idx in it.combinations_with_replacement(range(u.shape[1]),j)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def err(x,y):
 r=y-x@fit(x,y); h=np.sum(np.linalg.qr(x,mode='reduced')[0]**2,axis=1); return np.mean(abs(r)),np.mean(abs(r/np.maximum(1-h[:,None] if y.ndim>1 else 1-h,1e-8)))
for k in ['panel_b','panel_c']:
 a=s[k];u=a['u'];
 for h,v in a.items():
  if h=='u':continue
  print(k,h,'aff',err(aff(u),v),'quad',err(poly(u,2),v)); print(np.round(fit(poly(u,2),v),3))
w=np.asarray(o.get('w')); print('w',w.shape,w[:3]); print('traj SVD',np.linalg.svd(tr['traj'],compute_uv=False));
if w.ndim==2:
 for u in (tr['x'],w[:len(tr['x'])]):
  for d in range(1,4):print('heads',d,err(poly(u,d),tr['y']),err(poly(u,d),tr['traj']))
