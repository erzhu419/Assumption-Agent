import importlib.util, pickle, numpy as np
from pathlib import Path
from scipy.optimize import linprog
root=Path('/holdout_task')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=load('f',root/'_frozen_nodes/node_00.py');m=load('m',root/'node_01.py');s=pickle.load(open(root/'public_batch.pkl','rb'))[0]['study']; n=len(s['train']['y']); w=f.run_node({'__input__':s})['w']; _,v,q,_=m._components(s,w); v,q=v[:n],q[:n]; z=np.c_[s['train']['y'],s['train']['traj']]; folds=[np.arange(n-20,n)]+list(np.array_split(np.arange(n),4)); rng=np.random.default_rng(623); folds+=list(np.array_split(rng.permutation(n),4)); data=[]
for h in folds:
 tr=np.setdiff1d(np.arange(n),h); order=np.r_[tr,h]; p=m._kernel_predictions(v[order],q[order],z[tr],m._CONFIG);data.append(p-z[h]);
# Choose compact mixtures by absolute-error minimax across blocked and shuffled validation slices.
def optimize(errs,limit):
 k=len(m._CONFIG); blocks=[e.reshape(k,-1).T for e in errs]; N=sum(map(len,blocks)); A=np.zeros((2*N+len(blocks),k+N+1)); b=np.zeros(len(A)); off=0
 for j,E in enumerate(blocks):
  l=len(E); r=np.arange(off,off+l); A[r,:k]=E;A[r+N,:k]=-E; A[r,k+r]=-1; A[r+N,k+r]=-1; A[2*N+j,k+r]=1/l
  if j==0: b[2*N+j]=limit
  else:A[2*N+j,-1]=-1
  off+=l
 obj=np.r_[np.zeros(k+N),1.]; eq=np.r_[np.ones(k),np.zeros(N+1)][None,:]; sol=linprog(obj,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.],bounds=[(0,.85)]*k+[(0,None)]*(N+1),method='highs');return sol.x[:k] if sol.success else None
old=(m._Y_WEIGHTS,m._T_WEIGHTS); opts=[old];
for lim in [.036,.037,.0378,.038,.043,1.]:
 a=optimize([e[:,:,0] for e in data],lim);b=optimize([e[:,:,1:] for e in data],lim)
 if a is not None and b is not None:opts.append((a,b))
def validate(ab):
 a,b=ab; return np.array([[np.mean(abs(np.tensordot(a,e[:,:,0],axes=1))),np.mean(abs(np.tensordot(b,e[:,:,1:],axes=1)))] for e in data]);
for ab in opts: print('CV official / worst',validate(ab)[0],validate(ab).max(0))
def rank(ab):
 e=validate(ab);return (max(0.,float(max(e[0]))-.03795),float(max(e.max(0))))
a,b=min(opts,key=rank);print('SELECTED',validate((a,b)).round(6)); file=root/'node_01.py'; txt=file.read_text();import re
for name,w in [('_Y_WEIGHTS',a),('_T_WEIGHTS',b)]: txt=re.sub(r'^'+name+r' = .*$',name+' = '+repr(tuple(round(float(x),10) for x in w)),txt,flags=re.M)
file.write_text(txt)
