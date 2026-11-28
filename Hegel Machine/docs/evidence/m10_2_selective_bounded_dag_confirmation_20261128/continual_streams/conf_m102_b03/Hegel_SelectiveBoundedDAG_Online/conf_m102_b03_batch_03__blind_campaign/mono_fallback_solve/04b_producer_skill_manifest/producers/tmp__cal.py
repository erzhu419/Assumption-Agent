from explore import *
from scipy.optimize import least_squares
p=s['panel_a'];u=p['u'];v=p['v'];_,e=fit(one(np.c_[u,pair(u)]),v);g=e<1
for j in range(4):
 a=u[g,j]; b=u[g,(j+1)%4]; w=v[g,j]; F=one(np.c_[a,b,a*b]); E=w-F@ls(F,w); print('A',j,ls(F,w),np.std(E))
 D=[(f'{i}^{k}',u[g,i]**k) for i in range(4) for k in range(2,7)];D += [(str(ii),np.prod(u[g][:,ii],axis=1)) for k in range(2,5) for ii in itertools.combinations_with_replacement(range(4),k)];D += [(f'{nm}({k}*{i})',fn(k*u[g,i])) for nm,fn in [('sin',np.sin),('cos',np.cos),('soft',lambda x:x/(1+abs(x))),('tanh',np.tanh),('rat',lambda x:1/(1+x*x))] for i in range(4) for k in [.5,1,1.5,2,2.5,3,4]];D +=[(str((i,k,nm)),fn(u[g,i]-u[g,k])) for i,k in itertools.combinations(range(4),2) for nm,fn in [('cos',np.cos),('sin',np.sin),('cube',lambda x:x**3),('sq',lambda x:x*x)]]
 def trial(F):
  V=np.stack([d for _,d in D],axis=1); V-=F@ls(F,V); y=w-F@ls(F,w); n=np.linalg.norm(V,axis=0); sc=np.abs(y@V)/np.maximum(n,1e-10);ids=np.argsort(sc)[-7:][::-1];print([(D[i][0],round(np.sqrt(max(0,np.dot(y,y)-sc[i]**2)/len(w)),6)) for i in ids]);return np.c_[F,D[ids[0]][1]]
 trial(trial(F))
 for typ,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda z:1/(1+z*z))]:
  def phi(P):return np.c_[F,fun(u[g]@P[:4]+P[4])]
  def err(P):
   M=phi(P);return M@ls(M,w)-w
  starts=[np.r_[np.eye(4)[i]*k,phase] for i in range(4) for k in [1.,2.5] for phase in [0.,.4]]
  z=min((least_squares(err,P,max_nfev=160) for P in starts),key=lambda q:q.cost); print(typ,z.x, np.std(z.fun),ls(phi(z.x),w))
