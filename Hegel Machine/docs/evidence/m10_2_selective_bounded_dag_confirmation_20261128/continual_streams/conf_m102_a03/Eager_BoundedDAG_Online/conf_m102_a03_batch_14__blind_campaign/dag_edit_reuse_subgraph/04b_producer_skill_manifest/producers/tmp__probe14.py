import os
os.chdir('/holdout_task')
import pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; enc=runpy.run_path('_frozen_nodes/node_00.py')['run_node']({'__input__':s}); print('encoded',[(k,np.shape(v)) for k,v in enc.items()]);print('eval times',s['eval_times'])
def fit(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def powers(x,d):
 import itertools
 return np.column_stack([np.prod(x[:,inds],axis=1) for inds in itertools.combinations_with_replacement(range(x.shape[1]),d)])
for key in ('panel_b','panel_c'):
 p=s[key];x=p['u'];a=aff(x)
 for k,y in p.items():
  if k=='u':continue
  print(key,k,'affine',np.round(fit(a,y),4));print('row residual affine',np.round(np.linalg.norm(y-a@fit(a,y),axis=1),5))
  for fn,f in [('squares',lambda x,w:x*x),('quad',lambda x,w:powers(x,2)),('cube',lambda x,w:powers(x,3)),('tanh',lambda x,w:np.tanh(w*x)),('sin',lambda x,w:np.sin(w*x)),('cos',lambda x,w:np.cos(w*x)),('rational',lambda x,w:x/(1+w*x*x)),('norm',lambda x,w:x*np.sum(x*x,axis=1)[:,None])]:
   def loss(w,ret=False):
    D=np.c_[a,f(x,w)]; C=fit(D,y); e=y-D@C; lev=np.sum(D*np.linalg.pinv(D).T,axis=1); return (np.sqrt(np.mean(e*e)), np.sqrt(np.mean((e/np.maximum(1e-7,1-lev[:,None]))**2)),C) if ret else np.mean(e*e)
   opt=minimize_scalar(loss,bounds=(.25,4.5),method='bounded'); tr,cv,C=loss(opt.x,True); print(fn,round(opt.x,5),round(tr,6),round(cv,6),np.round(C[5:],3).tolist() if tr<.006 and len(C)<12 else '')
T=s['train']['traj'];print('traj singular',np.linalg.svd(T,compute_uv=False));print('traj successive slopes',np.round((T[:,1:]/T[:,:-1])[:5],4));print('last traj',T[:3])
