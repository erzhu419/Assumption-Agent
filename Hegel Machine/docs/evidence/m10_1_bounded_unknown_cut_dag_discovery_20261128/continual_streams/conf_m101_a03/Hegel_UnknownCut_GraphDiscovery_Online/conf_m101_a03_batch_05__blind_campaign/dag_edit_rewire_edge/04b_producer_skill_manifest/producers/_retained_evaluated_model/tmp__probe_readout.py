import pickle, numpy as np
from itertools import combinations_with_replacement
from pathlib import Path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a={}; b={}
exec(Path('/holdout_task/_frozen_nodes/node_00.py').read_text(),a); exec(Path('/holdout_task/_frozen_nodes/node_01.py').read_text(),b)
c=a['run_node']({'__input__':s}); d=b['run_node']({'__input__':s,'node_00':c}); x,z,y,T=c['x_tr_cal'],d['z_tr'],d['y'],d['traj']; Y=np.column_stack((y,T)); n=len(y)
print('warp',b['_fit_panel'](s['panel_b']['u'],s['panel_b']['v'])[:2]); print('trajectory singular values',np.linalg.svd(T,compute_uv=False)); print('label correlation across time',np.corrcoef(Y.T)[0].round(4)); print('trajectory samples',T[:8].round(4)); print('mean',T.mean(0).round(5))
folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(120).permutation(n),5))
def poly(q,k):
 return np.column_stack([np.prod(q[:,j],axis=1) for j in combinations_with_replacement(range(q.shape[1]),k)])
def score(name,F):
 F=np.column_stack((np.ones(n),F)); E=[]
 for v in folds:
  t=np.setdiff1d(np.arange(n),v); E.append(np.abs(F[v]@np.linalg.lstsq(F[t],Y[t],rcond=None)[0]-Y[v]).mean(0))
 E=np.array(E); print(name,F.shape[1], 'public/worst y',np.round([E[0,0],max(E[:,0])],5),'public/worst t',np.round([E[0,1:].mean(),E[:,1:].mean(1).max()],5),'time public',E[0,1:].round(4))
for name,q in [('x',x),('z',z),('xz',np.c_[x,z])]:
 score(name,q); score(name+' quadratic',np.c_[q,poly(q,2)]); score(name+' diagonal square',np.c_[q,q*q]); score(name+' diagonal cubic',np.c_[q,q*q,q**3]); score(name+' harmonic',np.c_[q,np.sin(q),np.cos(q)]); score(name+' harmonic 2',np.c_[q,np.sin(2*q),np.cos(2*q)]); score(name+' tanh',np.c_[q,np.tanh(q),q*q]);
 if q.shape[1]==4: score(name+' cubic',np.c_[q,poly(q,2),poly(q,3)])
score('xz with same-stage quadratic',np.c_[x,z,poly(x,2),poly(z,2)]); score('xz with calibrated quadratic',np.c_[x,z,poly(x,2)]); score('xz with operator quadratic',np.c_[x,z,poly(z,2)])
