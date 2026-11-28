import numpy as np, pickle, itertools as it, sys, importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
spec=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
q=m.run_node({'__input__':s})['q_tr'];x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];u=s['panel_c']['u'];v1=s['panel_c']['v1'];v2=s['panel_c']['v2']
def poly(z,k=2):
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,js],axis=1) for i in range(1,k+1) for js in it.combinations_with_replacement(range(z.shape[1]),i)])
def report(A,Y,cv=False):
 if cv:
  splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(72) for i in range(5)]]
  es=[np.abs(A[j]@np.linalg.lstsq(A[i],Y[i],rcond=None)[0]-Y[j]).mean() for i,j in splits];return np.round([es[0],np.mean(es),max(es)],5)
 Q=np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)];r=Y-Q@(Q.T@Y);return np.round([np.abs(r).mean(),np.abs(r/(1-(Q*Q).sum(1)[:,None])).mean(),*np.sqrt(np.mean(r*r,axis=0))],6)
for lbl,A in [('lin',poly(u,1)),('quad',poly(u)),('lin+tanh',np.c_[poly(u,1),np.tanh(u)]),('lin+sin',np.c_[poly(u,1),np.sin(u)]),('lin+abs',np.c_[poly(u,1),np.abs(u)]),('quad+tanh',np.c_[poly(u),np.tanh(u)]),('square',np.c_[poly(u,1),u*u]),('no-linear tanh',np.c_[np.ones(len(u)),np.tanh(u)])]:print('panel',lbl,'v1',report(A,v1),'v2',report(A,v2))
for name,z in [('x',x),('q',q)]:
 for lbl,A in [('lin',poly(z,1)),('quad',poly(z)),('cub',poly(z,3)),('sin',np.c_[poly(z,2),np.sin(z),np.cos(z)]),('tanh',np.c_[poly(z,2),np.tanh(z)]),('tanh-only',poly(np.tanh(z),1))]: print(name,lbl,'y',report(A,y,True),'T',report(A,T,True))
for P in [v1,v2]:
 for func in [np.tanh,np.sin,np.cos,np.exp]:
  def design(z,w):return np.c_[poly(z,1),func(z*w[:4]+w[4:])]
  def err(w):
   A=design(u,w); return (A@np.linalg.lstsq(A,P,rcond=None)[0]-P).ravel()
  fit=least_squares(err,np.r_[np.ones(4),np.zeros(4)],max_nfev=300); print('cal-opt',func.__name__,np.round(fit.x,4),report(design(u,fit.x),P))
print('correlations q, y,T modes',np.round(np.corrcoef(np.c_[q,y,np.linalg.svd(T,full_matrices=False)[0][:,:2]].T),3))
