import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): v=runpy.run_path('/tmp/chamber_structure.py')
p,l,a,h,t,dt,Y,g,align=(v[k] for k in ('p','l','a','h','t','dt','Y','g','align'))
np.set_printoptions(precision=6,suppress=True)
X=align([g]); r=h[2]+X@h[3:]; M=np.isnan(g['coupon_features_raw']); clean=~M.any(1); labs=g['lab_id']; B=np.c_[np.ones(len(l)),r,np.eye(4)[labs][:,1:],M[:,M.any(0)]]
lin=np.r_[t,1.]; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
def F(q): return np.r_[-np.expm1(-t/q),1/q]
def G(q): return l-p@(np.exp(-tail/q)*-np.expm1(-dt/q))
def fit(mode,ids):
 def calc(u,out=False):
  z=np.exp(u); q=a if mode.startswith('intrinsic') else z[:2]; qd=z[-1] if 'driftfit' in mode else 1.; f=F(qd)
  terms=[G(q[0])[:,None]*F(q[0]),G(q[1])[:,None]*F(q[1]),l[:,None]*lin]
  if 'memfree' in mode: terms=[x[:,None]*F(s) for x in np.c_[l,p].T for s in q]+[l[:,None]*lin]
  if 'separate' in mode: terms[:2]=[G(s)[:,None]*F(w) for s,w in zip(z[2:4],q)]
  D=np.stack(terms+[x[:,None]*f for x in B.T],axis=-1); k=D.shape[-1]; c=np.linalg.lstsq(D[ids].reshape(-1,k),Y[ids].ravel(),rcond=None)[0]; e=D@c-Y
  return (q,qd,z,c,e) if out else e[ids].ravel()
 seed=[a[0],a[1],*(a if 'separate' in mode else []),*([1.] if 'driftfit' in mode else [])] if not mode.startswith('intrinsic') else ([1.] if 'driftfit' in mode else [])
 if not seed:return calc([],True)
 opts=[least_squares(calc,np.log(z),bounds=(np.log(.08),np.log(25)),max_nfev=110) for z in [seed,([.65,1.9]+seed[2:]) if len(seed)>=2 else seed]]
 return calc(min(opts,key=lambda s:s.cost).x,True)
n=np.arange(len(l))
for mode in ['intrinsic','intrinsic-driftfit','shared','shared-driftfit','separate','separate-driftfit','memfree','memfree-driftfit']:
 q,qd,z,c,e=fit(mode,n); E=np.array([fit(mode,n[n!=i])[-1][i] for i in n]); print(mode,q,qd,z,c,'RMSE',np.mean(e**2)**.5,'OOF clean by horizon',np.max(np.mean(E[clean,:-1]**2,axis=0)**.5),'rate',np.mean(E[clean,-1]**2)**.5,'all',np.max(np.mean(E**2,axis=0)**.5))
 if mode=='memfree-driftfit': print(c[:14].reshape(-1,2))
print('intrinsic',a,h)
