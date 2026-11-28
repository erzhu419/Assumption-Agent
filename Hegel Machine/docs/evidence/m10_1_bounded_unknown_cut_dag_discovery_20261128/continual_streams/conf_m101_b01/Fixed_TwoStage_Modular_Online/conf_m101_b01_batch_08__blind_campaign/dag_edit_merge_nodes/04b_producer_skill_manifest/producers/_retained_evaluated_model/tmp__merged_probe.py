import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
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
 for e,n,t,f in sorted(opts,key=lambda r:r[0]): print('B law',n,e,t)
 _,name,t,f=min(opts,key=lambda r:r[0]); return lambda z:np.c_[ones(z),f(z,t)]@ls(np.c_[ones(u),f(u,t)],v)
def c_fit(p):
 u=p['u']; v1=p['v1']; v2=p['v2'];
 def f1(z): return np.c_[ones(z),z*z[:,J]]
 def phi(z,t): return z+t[0]*z*z+t[1]*(1-np.cos(t[2]*z))
 def f2(z,t):return np.c_[ones(z),phi(z,t)]
 def res(t):
  X=f2(u,t); return (X@ls(X,v2)-v2).ravel()
 t=least_squares(res,[.35,.06,2.5],max_nfev=300).x; print('C2 law',t,'rmse',np.sqrt(np.mean(res(t)**2)))
 c1=ls(f1(u),v1); c2=ls(f2(u,t),v2); return lambda z:f1(z)@c1,lambda z:f2(z,t)@c2
if __name__=='__main__':
 a=a_fit(S['panel_a']); print('A coefs',a); b=b_fit(S['panel_b']); c,d=c_fit(S['panel_c']); x=S['train']['x']; z,r=a_inv(x,a); print('A inversion',r,z.min(0),z.max(0)); print('A test inversion',a_inv(S['test']['x'],a)[1]); Y=np.c_[S['train']['y'],S['train']['traj']];
 def loo(F):
  F=np.c_[ones(F),F]; P=np.linalg.pinv(F); R=(Y-F@(P@Y))/(1-np.sum(F*P.T,axis=1)[:,None]); return np.round([np.mean(abs(R[:,0])),np.mean(abs(R[:,1:])),np.max(np.mean(abs(R[:,1:]),axis=0))],5)
 for label,u in [('raw',x),('latent',z)]:
  fs=[u,b(u),c(u),d(u)]; F=np.column_stack(fs); print(label,'blocks',[(i,loo(f)) for i,f in enumerate(fs)],'all',loo(F)); print(label,'all+raw/latent',loo(np.c_[F,x if label=='latent' else z])); print('quadratic in u',loo(np.c_[F,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]])); print('squares and tanh',loo(np.c_[F,F**2,np.tanh(F)])); print('outputs against blocks',np.round(ls(np.c_[ones(u),F],Y)[:,[0,1,-1]],3))
