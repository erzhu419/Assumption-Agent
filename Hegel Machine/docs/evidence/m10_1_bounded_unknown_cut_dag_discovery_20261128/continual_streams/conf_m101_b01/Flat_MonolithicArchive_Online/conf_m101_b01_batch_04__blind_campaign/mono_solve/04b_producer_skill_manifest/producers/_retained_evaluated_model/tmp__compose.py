from shape import *
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; ids=np.arange(len(u)); functions={
'soft':(lambda x,k:x/(1+k*abs(x)),(.01,5)), 'slog':(lambda x,k:np.sign(x)*np.log1p(k*abs(x)),(.01,5)), 'soft2':(lambda x,k:x/(1+k*abs(x))**2,(.01,5)), 'atan':(lambda x,k:np.arctan(k*x),(.1,8)), 'softplus':(lambda x,k:np.logaddexp(0,k*x),(.1,5)), 'cos':(lambda x,k:np.cos(k*x),(.1,5)), 'cosh':(lambda x,k:np.logaddexp(k*x,-k*x),(.1,5)), 'exp':(lambda x,k:np.exp(k*x),(.1,3)), 'tanh':(lambda x,k:np.tanh(k*x),(.1,5))}
Z={'b':b,'x':x}; out={}
for key in ('v1','v2'):
 v=p[key]; scores=[]
 for name,(fn,bounds) in functions.items():
  for kind in ('only','raw','q'):
   def d(z,k):
    h=fn(z,k)
    return aff(h) if kind=='only' else np.c_[aff(z) if kind=='raw' else poly(z),h]
   def err(k,loo=False):
    A=d(u,k); w=ls(A,v); r=A@w-v
    if loo: r=r/np.maximum(1-np.sum(A*np.linalg.pinv(A).T,axis=1),.001)[:,None]
    return np.sqrt(np.mean(r*r))
   opt=minimize_scalar(err,bounds=bounds,method='bounded'); k=opt.x; w=ls(d(u,k),v); scores.append((err(k,True),opt.fun,name,kind,k,w))
   for src,z in [('b',b),('x',x)]: Z[key+name+kind+src]=d(z,k)@w
 for loo,r,na,ki,k,w in sorted(scores,key=lambda q:q[0])[:7]: print(key,loo,r,na,ki,k,np.round(w,3) if loo<.0002 else '',flush=True)
 out[key]=sorted(scores,key=lambda q:q[0])[:3]

def cv(D,v,reg):
 val=[]
 for i,j in folds:
  scale=np.maximum(np.std(D[i],axis=0),.02); scale[0]=1
  A=D/scale; w=np.linalg.solve(A[i].T@A[i]+reg*np.eye(A.shape[1]),A[i].T@v[i]); val.append(np.mean(abs(A[j]@w-v[j])))
 return val
for target in [y,T]:
 S=[]
 def add(n,D):
  for reg in (.00001,.003,.04,.3):
   err=cv(D,target,reg); S.append((max(err),err,n,reg))
 for n,z in Z.items():
  if n in ('b','x') or any(n.startswith(k+a[2]+a[3]) for k in out for a in out[k]):
   for deg in (1,2,3): add((n,deg),poly(z,deg))
   h=np.arctan(1.7*z); add((n,'atan'),poly(h)); add((n,'q+b'),np.c_[poly(z),b]); add((n,'q+aB'),np.c_[poly(z),np.arctan(2*b)]); add((n,'quadAll'),poly(np.c_[z,b],2))
 for a in out['v1']:
  for c in out['v2']:
   for j in ('x','b'):
    l=Z['v1'+a[2]+a[3]+j]; r=Z['v2'+c[2]+c[3]+j]; k=(a[2:4],c[2:4],j)
    add((k,'bothQ'),np.c_[poly(l),poly(r)[:,1:]]); add((k,'crossQ'),poly(np.c_[l,r],2)); add((k,'skip'),np.c_[poly(l),aff(r)[:,1:],b]); add((k,'separate'),np.c_[poly(l),poly(r)[:,1:],poly(np.arctan(2*b))[:,1:]])
 print('best',target.shape,sorted(S)[:12],flush=True)
