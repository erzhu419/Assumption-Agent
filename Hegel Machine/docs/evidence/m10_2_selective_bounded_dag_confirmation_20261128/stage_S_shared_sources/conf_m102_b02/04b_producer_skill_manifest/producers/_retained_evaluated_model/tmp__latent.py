exec(open('/tmp/routes.py').read().split('# Rank')[0])
from scipy.optimize import least_squares
from itertools import combinations
z=(x-A[0])@np.linalg.inv(A[1:]); z=col(z)@B
_,_,V=np.linalg.svd(T[:52],full_matrices=False); V=V[:2]; Y=np.c_[y,T@V.T]
rng=np.random.default_rng(131)
for tr,te in [(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]:
 m=Y[tr].mean(0); sc=Y[tr].std(0); G=(Y-m)/sc
 def fv(p,U,kind):
  q=U@p[:,:4].T+p[:,4]; h=np.tanh(q)
  if kind==0:return col(np.c_[U,h,h*h])
  if kind==1:return col(np.c_[h,h*h,*[h[:,i]*h[:,j] for i,j in combinations(range(h.shape[1]),2)]])
  if kind==2:return col(np.c_[U,h])
  if kind==3:return col(np.c_[U,h,*[U[:,i]*U[:,j] for i in range(4) for j in range(i,4)]])
  return col(np.c_[h,h*h,U])
 for kind in range(4):
  for k in (3,4,5,6):
   out=[]
   for r in range(4):
    p=np.zeros((k,5)); p[:min(k,4),:4]=np.eye(4)[:min(k,4)]*(1.8 if r<2 else 0); p+=rng.normal(0,.13 if r<2 else 1,p.shape)
    p0=p.copy(); reg=.012
    def err(flat):
     P=flat.reshape(k,5); F=fv(P,z[tr],kind); c=ls(np.r_[F,.012*np.eye(F.shape[1])],np.r_[G[tr],np.zeros((F.shape[1],3))]); return np.r_[(F@c-G[tr]).ravel(),reg*(P-p0).ravel(),.012*c.ravel()]
    opt=least_squares(err,p.ravel(),max_nfev=95,ftol=1e-6); P=opt.x.reshape(k,5); F=fv(P,z,kind); c=ls(np.r_[F[tr],.012*np.eye(F.shape[1])],np.r_[G[tr],np.zeros((F.shape[1],3))]); E=(F[te]@c-G[te])*sc; out.append((round(np.linalg.norm(opt.fun),4),round(abs(E[:,0]).mean(),5),round(abs(E[:,1:]@V).mean(),5)))
   print(tr[0],kind,k,sorted(out),flush=True)
