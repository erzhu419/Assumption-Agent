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
from scipy.optimize import least_squares
pc=s['panel_c']; u=pc['u']; ops=[]
for key,func in [('v1',np.tanh),('v2',np.cos)]:
 V=pc[key]; candidates=[]
 for use_linear in [False,True]:
  for shifts in [False,True]:
   def ds(x,p):
    k=p[:4]; h=p[4:] if shifts else 0.; return np.c_[aff(x) if use_linear else np.ones(len(x)),func(x*k+h)]
   def res(p):
    D=ds(u,p);return (D@fit(D,V)-V).ravel()
   lo=[.15]*4+([-2]*4 if shifts else []);hi=[6]*4+([2]*4 if shifts else [])
   r=least_squares(res,[2.]*4+([0.]*4 if shifts else []),bounds=(lo,hi),max_nfev=260)
   D=ds(u,r.x); W=fit(D,V); print(key,use_linear,shifts,np.std(r.fun),r.x)
   candidates.append((np.std(r.fun),use_linear,shifts,r.x,W))
 err,linear,shift,p,W=min(candidates,key=lambda a:a[0]);
 def op(x,linear=linear,shift=shift,p=p,W=W,func=func):return np.c_[aff(x) if linear else np.ones(len(x)),func(x*p[:4]+(p[4:] if shift else 0))]@W
 ops.append(op)
print('LABEL AND TRAJECTORY MODEL COMPARISON'); rows=[]
a0=a.copy(); b0=z.copy(); worlds={'a':a0,'b':b0}
for name,w in list(worlds.items()):
 for j,op in enumerate(ops,1):
  worlds[f'{name}c{j}']=op(w); worlds[f'{name}c{j}b']=aff(op(w))@B
for name,w in worlds.items():
 g,h,j,k=w.T
 for label,D in [('aff',aff(w)),('pair',np.c_[aff(w),g*j,h*k]),('pair3',np.c_[aff(w),g*j,h*k,g*h*j,g*h*k,g*j*k,h*j*k,g*h*j*k]),('quad',poly(w,2)),('cubic',poly(w,3)),('base+op',np.c_[aff(b0),w]),('basequad+op',np.c_[poly(b0,2),w]),('basequad+opsq',np.c_[poly(b0,2),w,w*w])]:
  if np.linalg.matrix_rank(D,tol=1e-9)<D.shape[1]:continue
  q=metrics(D);rows.append((q[1],q[0],name,label))
 print(name,'traj',np.round(metrics(aff(w),t)[:2],5))
for r in sorted(rows)[:24]:print(r)
print('Panel C additive branches');
for name,w in [('a',a0),('b',b0)]:
 for d in [1,2,3]:
  for spec in ['direct','square','products','difference','sum']:
   v1,v2=[op(w) for op in ops]; extra={'direct':np.c_[v1,v2],'square':np.c_[v1,v2,v1*v1,v2*v2],'products':np.c_[v1,v2,v1*v2],'difference':v1-v2,'sum':v1+v2}[spec]; D=np.c_[poly(w,d),extra];print(name,d,spec,np.round(metrics(D)[:2],5))
