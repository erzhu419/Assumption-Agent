import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import null_space
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; n=len(x)
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(z,d):return np.column_stack([np.prod(z[:,i],axis=1) if i else np.ones(len(z)) for k in range(d+1) for i in combinations_with_replacement(range(z.shape[1]),k)])
a=s['panel_a']; u,v=a['u'],a['v']; A=aff(u); g=np.ones(len(u),bool)
for i in range(4):
 c=fit(A[g],v[g]); e=np.linalg.norm(v-A@c,axis=1);g=e<4*np.median(e)
a=(x-c[0])/np.diag(c[1:]); b=s['panel_b']; B=fit(aff(b['u']),b['v']); z=aff(a)@B
np.set_printoptions(precision=4,suppress=True)
def metrics(D,target=y):
 c=fit(D[:52],target[:52]); e=D@c-target
 return np.mean(abs(e[:52])),np.mean(abs(e[52:])),np.max(abs(e[52:])),c
for name,w in [('a',a),('b',z)]:
 print(name,'range',w.min(0),w.max(0))
 for d in range(1,5):
  m=metrics(poly(w,d));print('poly',d,m[:3],m[3] if d<=3 else '')
 print('traj affine',metrics(aff(w),t)[:3]);print('traj coefficients',fit(aff(w),t))
 print('paired',metrics(np.c_[aff(w),w[:,0]*w[:,2]+w[:,1]*w[:,3]])[:3]); print('coord correlations',np.corrcoef(np.c_[y,w].T)[0])
 for fun in [np.sin,np.tanh,np.exp,np.arctan]:
  for k in [.5,1,1.7,2.5]:
   print(fun.__name__,k, np.round(metrics(np.c_[np.ones(n),fun(k*w)])[:2],5), np.round(metrics(np.c_[poly(w,2),fun(k*w)])[:2],5))
pc=s['panel_c']; u=pc['u']; q=poly(u,2)
for key in ['v1','v2']:
 V=pc[key]; lin=aff(u); print(key,'quadratic coefficients',fit(q,V));
 for f in [np.sin,np.tanh,np.cos,np.arctan]:
  for k in [0.5,1.,1.7,2.]:
   D=np.c_[lin,f(k*u)];print(f.__name__,k,np.mean((D@fit(D,V)-V)**2)**.5,end='; ')
  print()
