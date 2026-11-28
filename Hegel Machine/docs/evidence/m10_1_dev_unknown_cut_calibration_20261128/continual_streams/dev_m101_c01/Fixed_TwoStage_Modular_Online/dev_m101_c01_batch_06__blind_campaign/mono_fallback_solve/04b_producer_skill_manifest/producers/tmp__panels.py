from probe import *
from scipy.optimize import least_squares

def affine(u,v): return np.linalg.lstsq(design(u,1),v,rcond=None)[0]
def fitc(u,v):
 D=design(u,1); Q=design(u,2); Q=Q[:,[0,1,2,3,4,5,9,12,14]]
 ops=[('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('softsign',lambda x:x/(1+np.abs(x))),('cubic',lambda x:x**3),('sat',lambda x:x/np.sqrt(1+x*x))]
 out=[]
 for j in range(4):
  vj=v[:,j]; a=np.linalg.lstsq(D,vj,rcond=None)[0]; fits=[]
  for name,g in ops:
   for mode in (0,1):
    def pred(p,U):
     w=design(U,1); return p[5]*g(w@p[:5])+(p[6] if mode==0 else w@p[6:])
    candidates=[]
    for scale in (.65,1.1,1.8,2.5):
     p=np.r_[a*scale,1.,np.zeros(1 if mode==0 else 5)]
     fit=least_squares(lambda p:pred(p,u)-vj,p,max_nfev=220,ftol=1e-9,gtol=1e-9);candidates.append(fit)
    fit=min(candidates,key=lambda f:f.cost); fits.append((np.mean(fit.fun**2)**.5,name,mode,fit.x,lambda U,p=fit.x,g=g,mode=mode: p[5]*g(design(U,1)@p[:5])+(p[6] if mode==0 else design(U,1)@p[6:])))
  out.append(sorted(fits,key=lambda a:a[0]));
 return out

def panel_features(verbose=False):
 a=S['panel_a']; A,_=robust(design(a['u'],1),a['v']); b=S['panel_b']; B=affine(b['u'],b['v']); c=S['panel_c']; fits=fitc(c['u'],c['v1']); u=c['u']; D=design(u,2)[:,[0,1,2,3,4,5,9,12,14]]; w=np.linalg.lstsq(D,c['v2'],rcond=None)[0];
 if verbose:
  for f in fits: print('C1',[(round(x[0],6),x[1],x[2],np.round(x[3],4)) for x in f[:3]])
  print('C2 diagonal quadratic',np.std(D@w-c['v2']),np.round(w.T,4));print('B singular',np.linalg.svd(B[1:],compute_uv=False))
 return A,B,lambda U:np.column_stack([f[0][-1](U) for f in fits]),lambda U:design(U,2)[:,[0,1,2,3,4,5,9,12,14]]@w

if __name__=='__main__':
 A,B,c1,c2=panel_features(True); x,y,z=[S['train'][k] for k in ('x','y','traj')]; t=S['eval_times']; D=design(x,2)
 B0=lambda U:design(U,1)@B; BA=lambda U: B0(design(U,1)@A); BAI=lambda U:B0((U-A[0])@np.linalg.inv(A[1:])); bi=lambda U:(U-B[0])@np.linalg.inv(B[1:]);
 rng=np.random.default_rng(42); folds=[(np.arange(56),np.arange(56,72))];
 for q in range(3):
  i=rng.permutation(len(x));folds.append((i[:56],i[56:]))
 for label,Q in [('raw',x),('A',design(x,1)@A),('Ainv',(x-A[0])@np.linalg.inv(A[1:])),('B',B0(x)),('BA',BA(x)),('BAI',BAI(x)),('Bi',bi(x))]:
  feats={'base':D,'c1':np.c_[D,c1(Q)],'c2+c1':np.c_[D,c1(Q),c2(Q)],'c1cross':np.c_[D,design(c1(Q),2)[:,1:],(Q[:,:,None]*c1(Q)[:,None,:]).reshape(-1,16)],'sin':np.c_[D,np.sin(Q),np.cos(Q)],'cubic':design(Q,3)}
  for name,F in feats.items():
   target=np.c_[y,z[:,-1]]; err=[]
   for i,j in folds:
    W=np.linalg.lstsq(F[i],target[i],rcond=1e-6)[0]; err.append(np.abs(F[j]@W-target[j]).mean(0))
   print(label,name,np.round(err[0],4),np.round(np.max(err,axis=0),4))
