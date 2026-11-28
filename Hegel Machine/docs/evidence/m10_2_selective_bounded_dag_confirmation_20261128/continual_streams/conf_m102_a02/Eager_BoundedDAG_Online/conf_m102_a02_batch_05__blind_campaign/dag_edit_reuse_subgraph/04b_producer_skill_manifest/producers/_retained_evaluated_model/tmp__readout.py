exec(open('/tmp/operators.py').read().split('P={}')[0])
from scipy.optimize import minimize_scalar
P={}; G={}
for j,key,n in [('panel_b','v','sin'),('panel_c','v1','atan'),('panel_c','v2','exp')]:
 p=s[j]; u,V=p['u'],p[key]; e,k,P[key]=fitop(u,V,n,shared=True); G[key]=lambda x,n=n,k=k:ops[n](x,k); D=np.c_[A(u),G[key](u)]; print(key,np.linalg.lstsq(D,V,rcond=None)[0].T)
def products(u,v,sym=False): return np.column_stack([u[:,i]*v[:,j] for i in range(4) for j in range(4) if not sym or i<=j])
def feature(z):
 b,c,d=(P[k](z) for k in ('v','v1','v2')); h,i,j=(G[k](z) for k in ('v','v1','v2')); return {'x':z,'b':b,'c':c,'d':d,'h':h,'i':i,'j':j,'bb':products(b,b,True),'cc':products(c,c,True),'dd':products(d,d,True),'xx':products(z,z,True),'hh':products(h,h,True),'ii':products(i,i,True),'jj':products(j,j,True),'bc':products(b,c),'bd':products(b,d),'cd':products(c,d),'bx':products(b,z),'cx':products(c,z),'dx':products(d,z),'xj':z*(j-1),'hj':h*(j-1),'x4':z**4,'xb':z*b,'xc':z*c,'xd':z*d,'bc0':b*c,'bd0':b*d,'cd0':c*d}
B=feature(z); base=np.ones((len(z),1));
def check(ks,lam=.003):
 D=np.c_[base,*[B[k] for k in ks]]; D[:,1:]=(D[:,1:]-D[:,1:].mean(0))/np.maximum(D[:,1:].std(0),1e-6); return cv(D,Y,lam)[0]
results=[]
for n in (2,3,4):
 for ks in cc(B,n):
  if sum(B[k].shape[1] for k in ks)>36 or not any(k in ks for k in ('b','c','h','i')): continue
  E=check(ks); results.append((max(E),E[0],ks))
print('max',sorted(results)[:12]); print('first',sorted(results,key=lambda r:r[1])[:12]);
for ks in [('b','bb'),('b','bb','d'),('b','bb','dd'),('b','c','d'),('b','c','d','bb'),('b','c','d','bc'),('b','c','d','bd'),('b','c','d','cd'),('h','hh','j'),('b','bb','x','xx')]+[a[2] for a in sorted(results)[:4]]:
 print('ridge',ks,[(l,np.round(check(ks,l),4)) for l in [0.00001,.001,.01,.03,.1,.3,1]])
# separable variable projection with structured interactions among panel outputs
for k in ('b','c','d','h','i'):
 linear=B[k]; D=np.c_[base,linear]; R=Y-D@np.linalg.lstsq(D,Y,rcond=None)[0]; scores=[]
 for name,F in B.items():
  F=F-D@np.linalg.lstsq(D,F,rcond=None)[0]; coef=np.linalg.lstsq(F,R,rcond=1e-7)[0]; scores.append((np.std((R-F@coef)[:,0]),name))
 print('residual',k,sorted(scores)[:8])
