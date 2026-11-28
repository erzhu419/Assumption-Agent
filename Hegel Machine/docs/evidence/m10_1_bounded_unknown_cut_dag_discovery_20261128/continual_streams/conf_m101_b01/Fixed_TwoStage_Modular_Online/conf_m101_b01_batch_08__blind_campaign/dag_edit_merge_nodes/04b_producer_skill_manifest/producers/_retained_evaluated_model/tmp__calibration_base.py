from scipy.optimize import minimize_scalar,least_squares
J=[1,2,3,0]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def ones(u): return np.ones(len(u))
def a_fit(p):
 u=p['u']; v=p['v']; z=u[:,J]; X=np.stack([np.ones_like(u),u,z,u*z,u**3],axis=-1); W=np.ones(len(u));
 for _ in range(20):
  c=np.stack([ls(X[:,j]*W[:,None],v[:,j]*W) for j in range(4)]); r=np.linalg.norm(np.einsum('nij,ij->ni',X,c)-v,axis=1); W=1/np.sqrt(1+(r/.12)**2)
 g=np.argsort(r)[:-3]; return np.stack([ls(X[g,j],v[g,j]) for j in range(4)])
def a_inv(x,c):
 b,l,m,n,p=c.T; u=(x-b)/l; k=np.arange(4)
 for _ in range(22):
  z=u[:,J]; r=b+l*u+m*z+n*u*z+p*u**3-x; jac=np.zeros((len(x),4,4)); jac[:,k,k]=l+n*z+3*p*u*u; jac[:,k,J]=m+n*u; u-=np.clip(np.linalg.solve(jac,r[...,None])[...,0],-.65,.65)
 return u,np.max(abs(r))
def b_fit(p):
 u,v=p['u'],p['v']; opts=[]
 funs={'tanh':lambda x,t:x+t[0]*np.tanh(t[1]*x),'rat':lambda x,t:x+t[0]*x/(1+t[1]*x*x),'atan':lambda x,t:x+t[0]*np.arctan(t[1]*x),'sin':lambda x,t:x+t[0]*np.sin(t[1]*x),'soft':lambda x,t:x+t[0]*x/np.sqrt(1+t[1]*x*x),'tanh3':lambda x,t:np.tanh(t[1]*x)+t[0]*x**3,'rat3':lambda x,t:x**3/(1+t[1]*x*x)+t[0]*x,'sinh':lambda x,t:x+t[0]*np.sinh(t[1]*x),'tanh+rat':lambda x,t:x+t[0]*np.tanh(t[1]*x)+t[2]*x/(1+t[3]*x*x)}
 for name,f in funs.items():
  def res(t):
   X=np.c_[ones(u),f(u,t)]; return (X@ls(X,v)-v).ravel()
  best=None
  for q in [.45,.85,1.45,2.1]:
   t0=[-.7,q] if name!='tanh+rat' else [-.6,q,-.15,1.]
   z=least_squares(res,t0,bounds=([-10,.12]+([-10,.12] if len(t0)>2 else []),[10,6]+([10,9] if len(t0)>2 else [])),max_nfev=400)
   e=np.sqrt(np.mean(z.fun**2))
   if best is None or e<best[0]: best=(e,name,z.x,f)
  opts.append(best)
 _,name,t,f=min(opts,key=lambda r:r[0]); return lambda z:np.c_[ones(z),f(z,t)]@ls(np.c_[ones(u),f(u,t)],v)
def c_fit(p):
 u=p['u']; v1=p['v1']; v2=p['v2'];
 def f1(z): return np.c_[ones(z),z*z[:,J]]
 def phi(z,t): return z+t[0]*z*z+t[1]*(1-np.cos(t[2]*z))
 def f2(z,t):return np.c_[ones(z),phi(z,t)]
 def res(t):
  X=f2(u,t); return (X@ls(X,v2)-v2).ravel()
 t=least_squares(res,[.35,.06,2.5],max_nfev=300).x
 c1=ls(f1(u),v1); c2=ls(f2(u,t),v2); return lambda z:f1(z)@c1,lambda z:f2(z,t)@c2
