import pickle, runpy, numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})
b=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']({'__input__':s,'node_00':a})
x,z,y,t=a['x_tr_cal'],b['z_tr'],b['y'],b['traj']
Y=np.column_stack([y,t]); n=len(y)
folds=[np.array_split(np.random.default_rng(seed).permutation(n),6) for seed in [0,1,2,3]]
def affine(q): return np.column_stack([np.ones(len(q)),q])
def poly(q,d): return np.column_stack([np.ones(len(q))]+[np.prod(q[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(q.shape[1]),k)])
def cv(f):
 e=[]
 for fs in folds:
  for v in fs:
   m=np.ones(n,bool);m[v]=False
   e.append(np.abs(Y[v]-f[v]@np.linalg.lstsq(f[m],Y[m],rcond=None)[0]).mean(0))
 err=np.array(e); full=np.abs(Y-f@np.linalg.lstsq(f,Y,rcond=None)[0]).mean(0)
 return np.round([full[0],full[1:].mean(),err[:,0].mean(),err[:,1:].mean(),err[:,0].max(),err[:,1:].mean(1).max()],5)
print('ranges',x.min(0),x.max(0),z.min(0),z.max(0)); print('operator',np.linalg.lstsq(affine(x),z,rcond=None)[0]);print('target singular values',np.linalg.svd(t,compute_uv=False));print('affine trajectory coefficients',np.linalg.lstsq(affine(z),t,rcond=None)[0]);
for nm,q in [('x',x),('z',z)]:
 for d in [1,2,3]: print(nm,d,cv(poly(q,d)))
 print(nm,'quad coefficient y',np.linalg.lstsq(poly(q,2),y,rcond=None)[0]);print(nm,'cubic coefficient y',np.linalg.lstsq(poly(q,3),y,rcond=None)[0]);
 for f in [.5,1,1.5,2,3]:
  extra=np.column_stack([np.sin(f*q),np.cos(f*q)])
  print(nm,'trig',f,cv(np.column_stack([poly(q,2),extra])))
base=affine(z)
for label,extra in [('pair xz',(x[:,:,None]*z[:,None,:]).reshape(n,-1)),('sinh z',np.sinh(z)),('tanh z',np.tanh(z)),('tanh x',np.tanh(x)),('xz diag',x*z),('sin z',np.sin(z)),('sin x',np.sin(x)),('sat z',z/(1+z*z)),('sat x',x/(1+x*x))]: print(label,cv(np.column_stack([base,extra])))
