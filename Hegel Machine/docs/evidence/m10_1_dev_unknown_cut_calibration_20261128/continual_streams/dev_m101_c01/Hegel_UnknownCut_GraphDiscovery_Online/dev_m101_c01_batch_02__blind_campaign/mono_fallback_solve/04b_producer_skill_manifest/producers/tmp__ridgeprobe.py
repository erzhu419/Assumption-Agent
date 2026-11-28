from structure import *
from scipy.optimize import least_squares
from scipy.linalg import qr,solve_triangular
rng=np.random.default_rng(829)
ids=np.arange(72); order=np.argsort(np.linalg.norm(x,axis=1)); splits=[(ids[:56],ids[56:])]+[(i[:56],i[56:]) for i in [rng.permutation(ids) for _ in range(4)]]+[(order[:56],order[56:])]

def ridge(U,Y,ix,fn,d,depth=1,starts=14):
 P=poly(U,d); ps=P[ix]; Q,R=qr(ps,mode='economic'); V=Y[ix]; RY=V-Q@(Q.T@V)
 if fn=='sin': transform=lambda u:np.column_stack([np.sin(u),np.cos(u)])
 if fn=='tanh': transform=np.tanh
 if fn=='rat': transform=lambda u:u/(1+u*u)
 if fn=='soft': transform=lambda u:u/np.sqrt(1+u*u)
 dim=4 if fn=='sin' else 5
 def enrich(T,p):return transform(T@p[:4]+(p[4] if dim==5 else 0)).reshape(len(T),-1)
 def res(p,ret=False):
  F=np.column_stack([enrich(U,pp) for pp in p.reshape(-1,dim)]); QF=Q.T@F[ix]; M=F[ix]-Q@QF
  w=np.linalg.lstsq(M,RY,rcond=None)[0]
  if ret:return P@solve_triangular(R,Q.T@V-QF@w)+F@w
  return (M@w-RY).ravel()
 local=np.random.default_rng(817)
 axes=np.r_[np.eye(4),local.normal(size=(starts,4))*.85]; init=[np.r_[a0,0] if dim==5 else a0 for a0 in axes]
 bds=([-3]*dim,[3]*dim)
 p=[]
 for k in range(depth):
  opts=[least_squares(res,np.r_[p,p0],bounds=(bds[0]*(k+1),bds[1]*(k+1)),max_nfev=65,ftol=3e-7) for p0 in init if np.max(abs(p0))<3]; best=min(opts,key=lambda o:o.cost);p=best.x
 return res(p,True),p,np.mean(abs(best.fun))

if __name__=='__main__':
 print('raw quadratic coefficients',np.linalg.lstsq(poly(x[:56]),np.c_[y,z][0:56],rcond=None)[0][:,[0,1,8]].T)
 for f in ('sin','tanh','rat','soft'):
  for d in (1,2):
   errs=[]
   for i,j in splits:
    p,k,e=ridge(xa,y,i,f,d); errs.append(np.mean(abs(p[j]-y[j])))
    if i[0]==0 and np.array_equal(i,ids[:56]): print('scalar',f,d,'params',np.round(k,3),'fit',round(e,5),'first',round(errs[-1],5))
   print('scalar CV',f,d,np.round(errs,5))
 for f in ('sin','tanh','rat'):
  for dep in (1,2):
   errs=[]
   for i,j in splits:
    p,k,e=ridge(xa,y,i,f,2,dep,8); errs.append(np.mean(abs(p[j]-y[j])))
    if np.array_equal(i,ids[:56]): print('depth',f,dep,'params',np.round(k,3),'fit',round(e,5))
   print('depth CV',f,dep,np.round(errs,5))
 for f in ('sin','tanh','rat'):
  epi=[]
  for i,j in splits:
   F=next(F for e,name,q,F in temporal(z[i]) if name=='step-wave'); c=np.linalg.lstsq(F,z.T,rcond=None)[0].T; Z=[]
   for l in range(2):
    p,k,e=ridge(xa,c[:,l],i,f,2,1,8);Z.append(p)
    if np.array_equal(i,ids[:56]): print('amp',l,f,'params',np.round(k,3),'fit',round(e,5))
   epi.append(np.mean(abs(np.array(Z).T[j]@F.T-z[j])))
  print('traj CV',f,np.round(epi,5))
