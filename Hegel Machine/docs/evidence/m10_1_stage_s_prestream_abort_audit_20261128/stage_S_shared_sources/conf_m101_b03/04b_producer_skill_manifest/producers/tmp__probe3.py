import pickle,numpy as n,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n.set_printoptions(precision=4,suppress=True,linewidth=150)
def ls(a,b): return n.linalg.lstsq(a,b,rcond=None)[0]
def pol(x,d): return n.column_stack([n.prod(x[:,i],axis=1) for j in range(d+1) for i in it.combinations_with_replacement(range(x.shape[1]),j)])
def aff(p,k='v'):
 u,v=p['u'],p[k];f=pol(u,1); b=ls(f,v);err=n.linalg.norm(f@b-v,axis=1);return ls(f[err < max(.001,2*n.median(err))],v[err < max(.001,2*n.median(err))])
a,b=aff(s['panel_a']),aff(s['panel_b']); print('B singular',n.linalg.svd(b[1:])[1]);p=s['panel_c'];u=p['u'];f=pol(u,1);q=pol(u,2)
for key in ('v1','v2'):
 v=p[key];r=v-f@ls(f,v);print(key,'residual spectra',n.linalg.svd(r)[1]);print('quadratic coefficients\n',ls(q,v).T)
 for i in range(4):
  g=[]
  for j in range(4):
   for k in range(j,4):
    for val,label in [(u[:,j]*u[:,k],'prod'),(n.abs(u[:,j]+u[:,k]),'abs+'),(n.abs(u[:,j]-u[:,k]),'abs-'),(n.sin(u[:,j]+u[:,k]),'sin+'),(n.sin(u[:,j]-u[:,k]),'sin-'),(n.tanh(u[:,j]+u[:,k]),'tanh+'),(n.tanh(u[:,j]-u[:,k]),'tanh-'),(u[:,j]**2*u[:,k],'cube1'),(u[:,j]*u[:,k]**2,'cube2')]: g.append((f'{label}{j}{k}',val))
  for idx in it.combinations(range(4),3):g.append((f'triple{idx}',n.prod(u[:,idx],axis=1)))
  ranked=[]
  for nm,h in g:
   X=n.c_[f,h];e=v[:,i]-X@ls(X,v[:,i]); ranked.append((n.sqrt(n.mean(e**2)),nm))
  print(i,sorted(ranked)[:8])
t=s['eval_times'];z=s['train']['traj'];sv,_,_=0,0,0
for typ in ['exp-exp','exp-t','exp-drive','exp-drive2','exp-const','exp-step']:
 def fun(p):
  e=lambda k:-n.expm1(-k*t)
  if typ=='exp-exp': return n.array([e(p[0]),e(p[1])]);
  if typ=='exp-t':return n.array([e(p[0]),t]);
  if typ=='exp-drive':return n.array([e(p[0]),t-e(p[0])/p[0]]);
  if typ=='exp-drive2':return n.array([e(p[0]),t*t-2*t/p[0]+2*e(p[0])/p[0]**2]);
  if typ=='exp-const':return n.array([e(p[0]),n.ones_like(t)]);
  if typ=='exp-step':return n.array([e(p[0]),t*n.exp(-p[0]*t)]);
 fit=least_squares(lambda p:(ls(fun(p).T,z.T).T@fun(p)-z).ravel(),[1.5,.4] if typ=='exp-exp' else [1.5],bounds=(.001,30)); print(typ,fit.x,n.std(fit.fun),n.mean(abs(fit.fun)))
fit=least_squares(lambda p:(ls((F:=n.array([-n.expm1(-p[0]*t),-n.expm1(-p[1]*t)])).T,z.T).T@F-z).ravel(),[1.5,.4],bounds=(.001,30));F=n.array([-n.expm1(-fit.x[0]*t),-n.expm1(-fit.x[1]*t)]);obs=n.c_[s['train']['y'],ls(F.T,z.T).T]
x=s['train']['x'];d=(x-a[0])/n.diag(a[1:]);G={'raw':x,'Ainv':d,'BAinv':pol(d,1)@b,'Braw':pol(x,1)@b};G.update({k+'Binv':(v-b[0])@n.linalg.inv(b[1:]) for k,v in list(G.items())[:2]});
for k,v in G.items():
 print(k,'range',v.min(0),v.max(0));print('linear coeff',ls(pol(v,1),obs).T)
 for fn in [n.tanh,n.sin,n.abs,lambda v:v**2,lambda v:v**3]:
  X=n.c_[pol(v,1),fn(v)]; print(fn.__name__,n.mean(abs(X[56:]@ls(X[:56],obs[:56])-obs[56:]),axis=0))
