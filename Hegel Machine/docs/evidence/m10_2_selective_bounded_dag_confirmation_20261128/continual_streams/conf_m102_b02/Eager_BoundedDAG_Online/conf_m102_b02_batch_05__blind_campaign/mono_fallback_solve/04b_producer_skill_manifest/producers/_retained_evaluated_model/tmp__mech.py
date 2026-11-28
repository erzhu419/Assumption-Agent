from discover import *
from scipy.optimize import least_squares
p=S['panel_b']; u,v=p['u'],p['v']; G=D(np.c_[u,erf(1.4913*u)]); W=fit(G,v); M=W[5:]; h=(v-W[0])@np.linalg.inv(M); print('mix',W[1:5]@np.linalg.inv(M)); print('u vs unmixed'); ii=np.argsort(u.ravel()); print(np.round(np.c_[u.ravel()[ii],h.ravel()[ii],(h-(erf(1.4913*u)-1.557*u)).ravel()[ii]],5));
funcs={'cubicsoft':lambda x,q:x**3/(1+q[0]*abs(x)**q[1]),'cubichinge':lambda x,q:x**3+q[1]*np.sign(x)*np.maximum(abs(x)-q[0],0)**q[2], 'power':lambda x,q:np.sign(x)*abs(x)**q[0], 'cubicatan':lambda x,q:x**3*np.arctan(q[0]*x*x+q[1]), 'cubiclim':lambda x,q:x**3/(1+q[0]*np.maximum(abs(x)-q[1],0)**q[2]),'cubicsqrt':lambda x,q:x**3/np.sqrt(1+q[0]*x**q[1])}
for n,f in funcs.items():
 for free in [0,1]:
  def G(x,q):return D(np.c_[x,f(x,q)]) if free else D(f(x,q))
  def res(q):
   A=G(u,q);return (A@fit(A,v)-v).ravel()
  lo=[.01,.01,.2] if n in ['cubichinge','cubiclim'] else [.01,.05,.2];
  if n=='cubichinge':lo[1]=-100
  if n=='cubicsqrt':continue
  r=min((least_squares(res,[a,b,c],bounds=(lo,[20,100,10]),max_nfev=220) for a in [.25,.6,1.4,2.75] for b in [.1,1,3] for c in [1,2,3]),key=lambda p:p.cost);print(n,free,r.x,np.std(r.fun))
