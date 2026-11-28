exec(open('/tmp/probe.py').read().split('for name,p')[0])
from itertools import combinations,combinations_with_replacement
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
ls=a._ls; bias=a._bias
u,v=(s['panel_a'][k] for k in ('u','v')); rr=np.linalg.norm(v-u,axis=1); keep=rr<np.median(rr)*4
print('outlier rows',np.where(~keep)[0],rr[~keep]);u,v=u[keep],v[keep]
print('raw range',x.min(0),x.max(0),'panel v range',v.min(0),v.max(0)); print('affine',ls(bias(u),v)); print('examples',np.c_[u,v][:16]);
for j in range(4):
 y=v[:,j]; uj=u[:,j]; other=[k for k in range(4) if k!=j]; print('\noutput',j)
 forms={'lin':lambda w:w,'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'atan':np.arctan,'soft':lambda w:w/(1+abs(w)),'exp':lambda w:np.expm1(-w*w),'rational':lambda w:w/(1+w*w),'asinh':np.arcsinh}
 tests=[]
 for nm,g in forms.items():
  for k in other:
   uk=u[:,k]
   for kind in range(10):
    def mat(p):
     w=g(p[0]*uj); h=g(p[0]*uk)
     ds=[uj*uk, uj*uk*uk, uj*uj*uk, w*uk, uj*h, g(p[-1]*(uj+uk)), g(p[-1]*(uj-uk)), g(p[-1]*uj*uk), uk*uk, np.tanh(p[-1]*(uj+uk))]
     return bias(np.c_[u, w,uj**2,uj**3,ds[kind]])
    def fun(p):
     f=mat(p);return f@ls(f,y)-y
    r=least_squares(fun,[1.1,1.15],bounds=(.08,5),max_nfev=40);tests.append((np.linalg.norm(r.fun)/len(u)**.5,nm,k,kind,r.x))
 print('best',sorted(tests,key=lambda t:t[0])[:12])
 for label,f in [('affine',bias(u)),('additive',bias(np.c_[u,u*u,u**3,u**4,np.tanh(1.5*u)])),('quadratic',bias(np.c_[u,*[u[:,k]*u[:,l] for k,l in combinations_with_replacement(range(4),2)]])),('cyclic cubic all',bias(np.c_[u,u*u,u**3,*[u[:,k]*u[:,l] for k,l in combinations(range(4),2)]]))]:
  r=y-f@ls(f,y); print(label, np.quantile(abs(r),[.5,.9,1]));
 print('corr of residual',np.corrcoef(np.c_[u,u*u,u**3,np.sin(2*u),np.cumprod(u,axis=1),y-bias(u)@ls(bias(u),y)].T)[-1,:-1])
