exec(open('/tmp/probe.py').read().split('rec=[]')[0])
from scipy.optimize import minimize
from itertools import combinations_with_replacement
from scipy.special import expit

def poly(z,d): return np.column_stack([np.prod(z[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(z.shape[1]),k)])
def metric(p,y):return np.mean(abs(p-y))
print('traj top temporal',v[:3]); print('telemetry',s['telemetry_vec'])
for name in ['x','z','bz','bx','c1z','c1b','c2z','c2b']:
 X=charts[name]; res=[]
 for j,target in enumerate(Y.T):
  for d in (2,3):
   F=poly(X,d)
   for kind in ('linear','tanh','rat','asinh','sin','den'):
    def infer(i,o):
     t=target[i]; c=t.mean(); r=np.max(abs(t-c)); h=np.sqrt(np.sum(X**2,axis=1));
     def pred(p):
      k=np.exp(p[-1]); a=F@p[:-1];
      if kind=='linear':return a
      if kind=='tanh':return c+r*k*np.tanh(a/(r*k))
      if kind=='asinh':return c+r*k*np.arcsinh(a/(r*k))
      if kind=='sin':return c+r*k*np.sin(a/(r*k))
      if kind=='rat':return c+a/np.sqrt(1+(a/(r*k))**2)
      return a/(1+k*h*h)
     p0=np.r_[fit(F[i],t if kind in ('linear','den') else t-c), .5]
     p=least_squares(lambda p: np.r_[pred(p)[i]-t, .004*p[:-1]],p0,bounds=(np.r_[np.full(F.shape[1],-np.inf),-6.],np.r_[np.full(F.shape[1],np.inf),6.]),max_nfev=70).x;return pred(p)[o],np.exp(p[-1]),np.std(pred(p)[i]-t)
    p,k,e=infer(*splits[0]); gate=metric(p,target[splits[0][1]]); res.append((gate,d,kind,k,e,infer))
  for e in sorted(res,key=lambda e:e[0])[:4]:print(name,j,e[:5])
# State contribution to scalar; true state in fit only, full CV also checked separately.
for name in ['x','z','bz','c1z','c1b','c2z','c2b']:
 X=charts[name]; Out=[]
 for degree in (1,2,3):
  for use in (False,True):
   for r in (.000001,.001,.02):
    FG=poly(np.c_[X,a] if use else X,degree); G=FG@FG.T+r*np.eye(len(X)); E=scores(G);Out.append((E[0,0],E[1:,0].mean(),degree,use,r))
 print('measured state shortcut',name,sorted(Out)[:3])
