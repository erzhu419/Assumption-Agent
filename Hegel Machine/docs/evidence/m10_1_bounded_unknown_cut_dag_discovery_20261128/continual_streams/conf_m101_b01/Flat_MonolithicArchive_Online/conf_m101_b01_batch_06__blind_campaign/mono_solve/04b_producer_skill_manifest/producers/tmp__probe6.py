import pickle, numpy as np
from itertools import combinations
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]; ij=np.triu_indices(4,1)
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def one(x):return np.ones((len(x),1))
def poly(x,k):
 from itertools import combinations_with_replacement as cr
 return np.column_stack([np.prod(x[:,p],axis=1) for i in range(k+1) for p in cr(range(x.shape[1]),i)])
def score(d):
 p=d[52:]@ls(d[:52],Y[:52]); e=abs(p-Y[52:]); return np.round([e[:,0].mean(),e[:,1:].mean()],5)
def a_design(x):return np.c_[one(x),x,x*x,x**3]
a=s['panel_a']; A=a_design(a['u']); keep=np.arange(len(A));
for i in range(6):
 w=ls(A[keep],a['v'][keep]); e=np.linalg.norm(A@w-a['v'],axis=1); keep=np.argsort(e)[:40]
print('A robust coeff',w.round(4),'errors',np.sort(e).round(4));
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k]; print(p,k)
 for name,fn in [('aff',lambda x:poly(x,1)),('square',lambda x:np.c_[poly(x,1),x*x]),('cross',lambda x:np.c_[one(x),x[:,ij[0]]*x[:,ij[1]]]),('quadratic',lambda x:poly(x,2)),('diag cubic',a_design),('erf',lambda x:np.c_[poly(x,1),erf(x)]),('cos',lambda x:np.c_[poly(x,1),np.cos(x)]),('sin',lambda x:np.c_[poly(x,1),np.sin(x)])]:
  d=fn(u); h=d@np.linalg.pinv(d); print(name, 'loo',round(np.mean(abs((h@v-v)/(1-h.diagonal()[:,None]))),6),'coeff' if name=='quadratic' else '',ls(d,v).round(3) if name=='quadratic' else '')
print('raw poly scores',[(k,score(poly(x,k))) for k in range(1,5)])
