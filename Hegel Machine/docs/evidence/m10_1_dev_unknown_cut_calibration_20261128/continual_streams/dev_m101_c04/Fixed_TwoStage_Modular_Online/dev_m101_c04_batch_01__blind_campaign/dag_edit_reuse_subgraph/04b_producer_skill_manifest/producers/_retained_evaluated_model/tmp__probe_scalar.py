exec(open('/tmp/probe_node01.py').read().split("for w in (u,x):")[0])
from scipy.optimize import least_squares
p=s['panel_c']; print('panel C rows'); print(np.c_[p['u'],p['v1'],p['v2']][:8]);
for a,b in [('u','v1'),('u','v2'),('v1','v2')]:
 for k in (1,2):
  A=poly(p[a],k); B=p[b]; f=np.linalg.lstsq(A,B,rcond=None)[0]; print(a,b,k,np.sqrt(np.mean((A@f-B)**2,axis=0)),f if k==1 else '')
A=poly(x,2); co=np.linalg.lstsq(A[:52],y[:52],rcond=None)[0]; Q=np.zeros((4,4));
for (i,j),a in zip(itertools.combinations_with_replacement(range(4),2),co[5:]): Q[i,j]=Q[j,i]=a/(1+(i!=j))
print('quadratic eigensystem',np.linalg.eigh(Q)); print('cubic coefficients',np.linalg.lstsq(poly(x,3)[:52],y[:52],rcond=None)[0][15:]);
X=x[:52]; Y=y[:52]; rng=np.random.default_rng(246)
acts={'sin':np.sin, 'cos':np.cos,'tanh':np.tanh,'gauss':lambda z:np.exp(-z*z),'rational':lambda z:1/(1+z*z),'cubic':lambda z:z**3,'abs':np.abs}
for degree in (1,2):
 A=poly(x,degree); D=A[:52]; proj=np.linalg.pinv(D); r=Y-D@(proj@Y)
 for name,act in acts.items():
  for n in (1,2):
   def feature(z,p):
    p=p.reshape(n,5); return act(z@p[:,:4].T+p[:,4]);
   def res(p,details=False):
    F=feature(X,p); R=F-D@(proj@F); w=np.linalg.lstsq(R,r,rcond=1e-9)[0]; err=R@w-r
    if details: return np.mean(abs(A@(proj@(Y-F@w))+feature(x,p)@w-y)[52:]),w
    return err
   candidates=[]
   for j in range(15):
    fit=least_squares(res,rng.normal(0,1,n*5),max_nfev=160); candidates.append((np.mean(abs(fit.fun)),fit.x))
   train,p=min(candidates,key=lambda c:c[0]); print(degree,name,n,round(train,6),res(p,True),p)
