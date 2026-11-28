import pickle,runpy,numpy as np
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
p='/holdout_task/'; s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; m=runpy.run_path(p+'node_00.py'); z=m['_calibrate'](s['panel_a'],s['train']['x']); y=s['train']['y']; t=s['train']['traj']; Y=np.c_[y,t]; V=np.linalg.svd(t,full_matrices=False)[2][:2]; T=np.c_[y,t@V.T]; n=len(z); folds=list(KFold(6,shuffle=True,random_state=42).split(z)); folds.append((np.argsort(np.max(abs(z),axis=1))[:-12],np.argsort(np.max(abs(z),axis=1))[-12:]));
def cv(F,lam):
 es=[]
 for a,b in folds:
  f=F[a]; B=np.linalg.solve(f.T@f+lam*np.diag(np.r_[0,np.ones(f.shape[1]-1)]),f.T@T[a]); q=F[b]@B; es.append(np.mean(abs(np.c_[q[:,0],q[:,1:]@V]-Y[b]),axis=0))
 es=np.array(es); return np.round(np.r_[es[:-1,0].mean(),es[:-1,1:].mean(),es[:-1,0].max(),es[:-1,1:].mean(1).max(),es[-1,0],es[-1,1:].mean()],4)
def poly(d):return np.column_stack([np.ones(n)]+[np.prod(z[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
D=np.c_[np.ones(n),z]; W=poly(2); print('latent affine',np.round(np.linalg.lstsq(D,T,rcond=None)[0],3)); print('quadratic coefficients',np.round(np.linalg.lstsq(W,T,rcond=None)[0],3));
for k,F in [('linear',D),('square',np.c_[D,z*z]),('poly2',W),('poly3',poly(3)),('diag3',np.c_[W,z**3]),('diag5',np.c_[W,z**3,z**4,z**5]),('sin',np.c_[W,np.sin(2*z)]),('tanh',np.c_[W,np.tanh(2*z)]),('atan',np.c_[W,np.arctan(2.4*z)])]:
 print(k, F.shape[1], [(l,cv(F,l).tolist()) for l in [.0003,.01,.1]])
for l in [.2,.45,.8,1.5,3.,7.]:
 for kind in ['rbf','matern']:
  r=cdist(z,z)/l; K=np.exp(-r*r/2) if kind=='rbf' else (1+r+r*r/3)*np.exp(-r); E,S=np.linalg.eigh(K); F=np.c_[W,S*np.sqrt(np.maximum(E,0))]; print(kind,l,cv(F,.0003))
