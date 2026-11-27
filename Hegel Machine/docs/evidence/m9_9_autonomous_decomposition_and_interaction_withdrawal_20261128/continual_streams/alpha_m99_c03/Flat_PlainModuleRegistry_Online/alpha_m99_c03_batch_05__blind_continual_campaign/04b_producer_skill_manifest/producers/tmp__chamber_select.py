from chamber_probe import *
from scipy.optimize import least_squares
past,H,F=inp_d; lab=d['lab_id']; miss=np.isnan(d['coupon_features_raw']).any(axis=1)
_,G=m._design(tau,inp_d,dt,t); z=H[:,None]-G[:,:2]*tau
U=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); B=np.c_[U,np.r_[t,1.]]
R=np.c_[yd,rd]-H[:,None]*(U@th[:2]); one=np.ones(len(F))
D={k:np.concatenate([d[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
forms={'joint':xd,'train_diag':xd2,'diag_only':m._features(d,d)[:len(F)]}
for name,groups in [('deployment',(tr,te)),('train',(tr,)),('test',(te,))]:
 ref=np.concatenate([g['ref_coupons'] for g in groups]); labs=np.concatenate([g['lab_id'] for g in groups]); X=d['coupon_features_raw'].copy()
 for g in np.unique(lab):
  panel=ref[labs==g]; X[lab==g]=(X[lab==g]-np.median(panel.mean(axis=0)))/np.median(panel.std(axis=0))
 forms[name]=np.where(np.isfinite(X),X,xd)

def shape(mask):
 p=np.linalg.svd(R[mask],full_matrices=False)[2][0]; p=B@np.linalg.lstsq(B,p,rcond=None)[0]; return p/p[-1]

def lin(M,a,k): return M@np.linalg.lstsq(M[k],a[k],rcond=None)[0]

def law(kind,r,a,k):
 q=z[:,1]
 if kind=='linear': return lin(np.c_[one,r],a,k),[]
 if kind=='rational': return lin(np.c_[one,r/(1+.5*q*q)],a,k),[]
 if kind=='bilinear': return lin(np.c_[one,r,q,r*q],a,k),[]
 if kind=='bilinear2': return lin(np.c_[one,r,z,r[:,None]*z],a,k),[]
 if kind=='rat_offset':
  def matrix(p): return np.c_[one,r/(1+p[0]*(q-p[1])**2),1/(1+p[0]*(q-p[1])**2)]
  fits=[least_squares(lambda p:(lin(matrix(p),a,k)-a)[k],p0,bounds=([.005,-1.5],[3,2])) for p0 in ([.3,0],[.5,-.5],[.5,.65])]; fit=min(fits,key=lambda f:f.cost); return lin(matrix(fit.x),a,k),fit.x
 if kind=='ratio_bilinear':
  def matrix(p): return np.c_[one,r,q,r*q]/(1+p[0]*q[:,None]**2)
  fit=least_squares(lambda p:(lin(matrix(p),a,k)-a)[k],[.4],bounds=([0],[3])); return lin(matrix(fit.x),a,k),fit.x
 if kind=='rank_bilinear':
  def pred(p): return (p[0]+p[1]*r)*(1+p[2]*q)
  fit=least_squares(lambda p:(pred(p)-a)[k],[-.07,1.3,-.25]); return pred(fit.x),fit.x

def stats(E):
 w=np.max([np.sqrt(np.mean(E[lab==g]**2,axis=0)) for g in np.unique(lab)],axis=0); return [round(v,5) for v in [max(w[:-1]),w[-1],np.sqrt(np.mean(E**2)),np.max(abs(E[miss]))]]
for name,X in forms.items():
 r=th[2]+X@th[3:]; allm=np.ones(len(R),bool); P=shape(allm); a=R@P/(P@P)
 for kind in ('linear','rational','bilinear','bilinear2','rat_offset','ratio_bilinear','rank_bilinear'):
  E=np.empty_like(R)
  for g in np.unique(lab):
   k=lab!=g; p=shape(k); aa=R@p/(p@p); pred,_=law(kind,r,aa,k); E[~k]=R[~k]-pred[~k,None]*p
  pred,par=law(kind,r,a,allm); print(name,kind,stats(E),'fit-amplitude',round(np.std(a-pred),5),np.round(par,4))
print('per-lab amplitude residual means, labeled linear baseline')
for name,X in forms.items():
 r=th[2]+X@th[3:]; pred,_=law('linear',r,a,allm); print(name,[round(np.mean((a-pred)[lab==g]),4) for g in np.unique(lab)])
