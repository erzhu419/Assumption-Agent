exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares, minimize_scalar
np.set_printoptions(precision=3,suppress=True)
pc=s['panel_c']; U,V,W=[pc[k] for k in ['u','v1','v2']]; B=s['panel_b']; kb,gb=m._fit_b(B['u'],B['v']); print('B k',kb)
def loo(D,T):
 q=np.linalg.svd(D,full_matrices=False)[0]; E=(T-q@(q.T@T))/(1-(q*q).sum(1)[:,None]);return np.sqrt(np.mean(E**2))
def generic(t,k):return [np.tanh(k*t),np.sin(k*t),t*(-np.expm1(-k*t*t)),t/(1+k*t*t),t*abs(t)]
for name,T in [('v1',V),('v2',W)]:
 for q,qname in [(U,'u'),(V,'v1'),(m._b_design(U,kb)@gb,'b(u)')]:
  if q is T:continue
  print(name,qname,'poly',*[round(loo(poly(q,d),T),5) for d in [1,2]])
  for d in [1,2]:
   for j in range(5):
    fun=lambda k:loo(np.c_[poly(q,d),generic(q,k)[j]],T)
    opt=minimize_scalar(fun,bounds=(.03,4),method='bounded'); print(d,j,round(opt.x,4),round(opt.fun,5),end='; ')
   print()
  for fun in [np.arctanh,np.arcsinh,np.arctan]:
   fo=lambda k:loo(poly(q,1),fun(T/k))*k
   opt=minimize_scalar(fo,bounds=(np.abs(T).max()+1e-5,5),method='bounded'); print(fun.__name__,round(opt.x,4),round(opt.fun,5),end='; ')
  print()
A=poly(z,1); Q=poly(z,2); R=Q[:,5:]; C=np.linalg.lstsq(Q,Y,rcond=None)[0]; E=Y-Q@C
print('linear residual svd directions',np.linalg.svd(Y-A@np.linalg.lstsq(A,Y,rcond=None)[0],full_matrices=False)[2][:3]);print('quadratic residual directions',np.linalg.svd(E,full_matrices=False)[2][:3]); print('linear coeff by time',C[:5]);
def probe(a,b,kind):
 D=A[a]; N=R[a]; P=np.linalg.qr(D)[0]; T=Y[a]-P@(P.T@Y[a]); q0=np.linalg.lstsq(np.c_[D,N],Y[a],rcond=None)[0][:,0];
 def feats(q,p):
  h=q@p[:-1]; k=p[-1]; h=np.clip(h,-6,6)
  return {'tanh':lambda:np.tanh(h),'sin':lambda:np.sin(h),'asinh':lambda:np.arcsinh(h),'exp':lambda:np.exp(h),'rational':lambda:h/(1+k*k*h*h),'signed_square':lambda:h*abs(h),'quad':lambda:h+k*h*h}[kind]()
 def fit(q,p):return np.c_[q[:,:5],feats(q,p)]
 def res(p):
  f=feats(Q[a],p); f-=P@(P.T@f);f/=max(1e-10,np.linalg.norm(f));return (T-np.outer(f,f@T)).ravel()
 best=min((least_squares(res,np.r_[k*q0,0.5],max_nfev=120) for k in [.3,1,2]),key=lambda t:t.cost)
 D=fit(Q[a],best.x); c=np.linalg.lstsq(D,Y[a],rcond=None)[0]; err=np.mean(abs(fit(Q[b],best.x)@c-Y[b]),axis=0);return best.cost,err[0],err[1:].mean(),best.x
for k in ['tanh','sin','asinh','exp','rational','signed_square','quad']:
 cost,ey,et,p=probe(*fs[0],k);print(k,round(cost,5),round(ey,5),round(et,5),p)
