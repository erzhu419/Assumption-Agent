import importlib.util, itertools as it, pickle
import numpy as np
from pathlib import Path
root=Path('/holdout_task')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
study=pickle.loads((root/'public_batch.pkl').read_bytes())[0]['study']
c=load('n0',root/'_frozen_nodes/node_00.py').run_node({'__input__':study})['c']
y=study['train']['y']; traj=study['train']['traj']; n=len(y)
A=np.column_stack((np.ones(n),c)); pairs=list(it.combinations_with_replacement(range(c.shape[1]),2)); Q=np.column_stack([c[:,i]*c[:,j] for i,j in pairs]); cross=[pairs.index(p) for p in [(0,2),(0,3),(1,2),(1,3)]]; off=[j for j,(p,q) in enumerate(pairs) if p!=q]; old=Q[:,cross[0]]+Q[:,cross[3]]
D={'affine':A,'archived':np.c_[A,old],'cross':np.c_[A,Q[:,cross]],'offdiag':np.c_[A,Q[:,off]],'quadratic':np.c_[A,Q], 'cubic':np.c_[A,Q,np.column_stack([np.prod(c[:,p],axis=1) for p in it.combinations_with_replacement(range(4),3)])]}
D.update({f'{pairs[p]} {sgn:+} {pairs[q]}':np.c_[A,Q[:,p]+sgn*Q[:,q]] for p,q in it.combinations(off,2) for sgn in [1,-1]})
D.update({str(pairs[p]):np.c_[A,Q[:,p]] for p in range(len(pairs))})
folds=[(np.arange(n-20),np.arange(n-20,n))]+[(np.setdiff1d(np.arange(n),te),te) for te in np.array_split(np.random.default_rng(117).permutation(n),6)]
def pred(d,target,tr,te): return d[te]@np.linalg.lstsq(d[tr],target[tr],rcond=None)[0]
def errors(d, target=y, splits=folds): return np.array([np.mean(np.abs(target[te]-pred(d,target,tr,te))) for tr,te in splits])
for key in ['affine','archived','cross','offdiag','quadratic','cubic']+sorted(set(D)-{'affine','archived','cross','offdiag','quadratic','cubic'},key=lambda k:errors(D[k])[1:].mean())[:10]:
 d=D[key]; e=errors(d); b=np.linalg.lstsq(d,y,rcond=None)[0]; print(key, 'holdout/oof/worst',np.round([e[0],e[1:].mean(),e.max()],6),'coefficients',b.round(5),'train',np.mean(np.abs(y-d@b)))
# Retest the discrete dictionary with motif selection restricted to each training fold.
keys=list(D)
for tr,te in folds:
 inn=[(np.setdiff1d(tr,va),va) for va in np.array_split(np.random.default_rng(213).permutation(tr),5)]
 key=min(keys,key=lambda k:errors(D[k],splits=inn).mean()); print('nested',key,round(np.mean(np.abs(y[te]-pred(D[key],y,tr,te))),6))
print('trajectory affine holdout/oof/worst',errors(A,traj)[0],errors(A,traj)[1:].mean(),errors(A,traj).max()); print('pairs',pairs)
p=study['panel_c']; u=np.c_[np.ones(len(p['u'])),p['u']]; print('panel C affine fits',[(k,np.mean(np.abs(u@np.linalg.lstsq(u,p[k],rcond=None)[0]-p[k]))) for k in ('v1','v2')])
