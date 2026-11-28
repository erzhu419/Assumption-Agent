import numpy as n,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n.set_printoptions(precision=4,suppress=True,linewidth=150)
def ls(a,b):return n.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,d):return n.column_stack([n.ones(len(x))]+[n.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
a=s['panel_a']; A=poly(a['u'],1); keep=n.ones(len(A),bool)
for _ in range(4):
 b=ls(A[keep],a['v'][keep]); r=n.linalg.norm(A@b-a['v'],axis=1);keep=r<max(.001,3*n.median(r))
z=n.linalg.solve(b[1:].T,(s['train']['x']-b[0]).T).T; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; Y=n.c_[y,T];
def temporal(q):return -n.expm1(-t[:,None]*n.exp(q)[None,:])
for fn in (temporal,lambda q:n.c_[temporal(q),t],lambda q:n.c_[temporal(q),n.ones(len(t))],lambda q:n.c_[temporal(q[:1]),n.sin(t*n.exp(q[1]))]):
 opt=least_squares(lambda q:(fn(q)@ls(fn(q),T.T)-T.T).ravel(),[0,1]); F=fn(opt.x); print('time',n.exp(opt.x), n.sqrt(n.mean(opt.fun**2)),ls(F,T.T)[:,:8]);
for key in ('panel_b','panel_c'):
 p=s[key]; x=p['u']; print(key,'u ranges',x.min(0),x.max(0)); print(n.column_stack(list(p.values()))[:10]);
 for k,v in p.items():
  if k=='u':continue
  L=poly(x,1); init=ls(L,v)
  for name,fun in [('tanh',n.tanh),('sin',n.sin),('asinh',n.arcsinh),('rational',lambda x:x/(1+x*x)),('atan',n.arctan)]:
   err=[]
   for i in range(4):
    def pred(q):return q[0]+q[1]*fun(L@q[2:]);
    f=least_squares(lambda q:pred(q)-v[:,i],n.r_[0,1,init[:,i]],max_nfev=200);err.append(n.std(f.fun))
   print(k,name,err)
  for d in (1,2):
   P=poly(x,d); c=ls(P,v); print(k,d,'coeff',c); w=poly(z,d)@c
   for dd in (1,2,3):
    J=poly(w,dd); print('embedding',dd,n.mean(abs(J[52:]@ls(J[:52],Y[:52])-Y[52:]),axis=0)[[0,1,8]])
