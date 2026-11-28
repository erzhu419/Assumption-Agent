import os, sys, pickle, runpy, numpy as np
from scipy.optimize import least_squares
from itertools import combinations
os.chdir('/holdout_task');sys.path.insert(0,os.getcwd());from node_01 import _coords,_aff,_fit,_quad,_panel
s=pickle.load(open('public_batch.pkl','rb'))[0]['study'];en=runpy.run_path('_frozen_nodes/node_00.py')['run_node']({'__input__':s});z=_coords(s,en['z']); print('z minmax',z.min(0),z.max(0)); T=s['train']['traj']; Y=np.c_[s['train']['y'],T]; n=52
v=_panel(z,s['panel_c'],'v1',np.tanh,[2.,2.27]);w=_panel(z,s['panel_c'],'v2',np.cos,[-.65,1.29]);
def met(D):
 P=D@_fit(D[:n],Y[:n]); E=abs(Y-P);return tuple(np.round([E[:n,0].mean(),E[:n,1:].mean(),E[n:,0].mean(),E[n:,1:].mean()],5))
q=_aff(z); c=_fit(q[:n],Y[:n]); E=Y-q@c; print('resid sv',np.linalg.svd(E[:n],compute_uv=False)); print('resid corr',np.corrcoef(E.T)[0]);print('yc',c[:,[0,-1]]);
for use in ('z','v','w','all'):
 for quad in (False,True):
  A=np.c_[q,_quad(z)] if quad else q; cols=[]; labels=[]
  def add(k,x):
   x=np.asarray(x);cols.append(x);labels.append(k)
  X={'z':z,'v':v,'w':w}; blocks=list(X) if use=='all' else [use];
  for k in blocks:
   x=X[k];
   for i in range(4):
    p=x[:,i]
    for f in [.5,1.25,2.15,2.9,4.,5.]:
     for nn,fn in [('tan',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda a:a/(1+a*a))]:add((k,i,nn,f),fn(f*p))
   for i,j in combinations(range(4),2):
    add((k,'prod',i,j),x[:,i]*x[:,j]);
    for sign in (-1,1):
     p=x[:,i]+sign*x[:,j];
     for f in [1.25,2.15,3.,4.]:
      for nn,fn in [('tan',np.tanh),('sin',np.sin),('cos',np.cos),('rat',lambda a:a/(1+a*a))]:add((k,i,j,sign,nn,f),fn(f*p))
   for inds in combinations(range(4),3):add((k,'triple',inds),np.prod(x[:,inds],axis=1))
  F=np.array(cols).T;remain=list(range(len(cols)));active=[]
  for step in range(8):
   d=A[:n];P=np.linalg.pinv(d);R=Y[:n]-d@(P@Y[:n]);G=F[:n,remain]-d@(P@F[:n,remain]);gain=((G.T@R)**2)@np.r_[1.,np.ones(8)/4]/np.maximum(np.sum(G*G,axis=0),1e-15); ix=remain.pop(np.argmax(gain));active.append(labels[ix]);A=np.c_[A,F[:,ix]];print(use,quad,step,labels[ix],met(A),flush=True)
