exec(open('/tmp/explore.py').read().split('for fn in')[0])
from scipy.optimize import minimize_scalar
from scipy.linalg import svdvals
rng=n.random.default_rng(7)
def fitval(A,V,cv=False):
 if not cv:return n.sqrt(n.mean((A@ls(A,V)-V)**2))
 return n.sqrt(n.mean([(A[i]@ls(n.delete(A,i,0),n.delete(V,i,0))-V[i])**2 for i in range(len(A))]))
for key in ('panel_b','panel_c'):
 p=s[key]; u=p['u']; print('\n',key)
 for k,v in p.items():
  if k=='u':continue
  print(k)
  L=poly(u,1)
  for name,A in [('aff',L),('sq',n.c_[L,u**2]),('cube',n.c_[L,u**2,u**3]),('cubicodd',n.c_[L,u**3]),('cross',poly(u,2))]:print(name,fitval(A,v),fitval(A,v,True))
  for name,f in [('sin',n.sin),('cos',n.cos),('tanh',n.tanh),('exp',n.exp),('rational',lambda x:x/(1+x*x))]:
   def fun(a):return fitval(n.c_[L,f(a*u)],v)
   opts=[minimize_scalar(fun,bounds=(j,j+.5),method='bounded') for j in n.arange(.05,6,.5)]; o=min(opts,key=lambda o:o.fun); print(name,round(o.x,5),round(o.fun,5),round(fitval(n.c_[L,f(o.x*u)],v,True),5))
  print('resid SV',svdvals(v-L@ls(L,v)))
  for j in range(4):
   target=v[:,j]; best=[]
   for name,f in [('tanh',n.tanh),('sin',n.sin),('cos',n.cos),('rational',lambda x:x/(1+x*x))]:
    def fun(q):
     D=n.c_[L,f(L@q)];return D@ls(D,target)-target
    opts=[least_squares(fun,rng.normal(0,.9,5),max_nfev=80) for _ in range(10)];o=min(opts,key=lambda o:n.linalg.norm(o.fun));best.append((name,round(n.std(o.fun),5),n.round(o.x,3)))
   print(j,best)
print('\ny/modes')
basis=n.linalg.svd(T[:52],full_matrices=False)[2][:2]; V=n.c_[y,T@basis.T];
L=poly(z,1)
for A in (L,n.c_[L,z*z],poly(z,2),poly(z,3)):
 c=ls(A[:52],V[:52]); print('coefs',c[:15]);print('CV',n.mean(abs(A[52:]@c-V[52:]),0))
for title,f in [('tanh',n.tanh),('sin',n.sin),('asinh',n.arcsinh),('power',lambda v:v**3)]:
 for pos in range(3):
  def ff(q,x):return n.c_[poly(x,1),f(poly(x,1)@q.reshape(-1,5).T)]
  def fun(q):
   D=ff(q,z[:52]);return D@ls(D,V[:52,pos])-V[:52,pos]
  print(title,pos,end=' ')
  for count in [1,2]:
   opts=[least_squares(fun,rng.normal(0,.9,5*count),max_nfev=120) for _ in range(12)];o=min(opts,key=lambda o:n.linalg.norm(o.fun)); D=ff(o.x,z);c=ls(D[:52],V[:52,pos]);print(round(n.std(o.fun),4),round(n.mean(abs(D[52:]@c-V[52:,pos])),4),o.x,end='; ')
  print()
