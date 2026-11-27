import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/chamber98.py').read())
from scipy.optimize import least_squares
Xraw=c['coupon_features_raw']; imiss=np.isnan(Xraw); V=np.c_[Y,2*R]; o=np.ones(len(H)); X=m._features(a,c)[n:]; Xall=m._features(a,{k:np.concatenate([b[k],c[k]]) for k in c if k in b})[-len(lab):]; raw0=np.where(imiss,0,Xraw)
print('material',tau,th); print('diagnostic lab, missing, raw features',np.c_[lab,imiss.sum(1),raw0].round(3)); print('diag/test keys',list(c),list(b)); print('train fit implementation'); import inspect; print(inspect.getsource(m._fit))
T=np.vstack([np.c_[-np.expm1(-t[:,None]/[.472,2.47]),t],np.r_[1/np.array([.472,2.47]),1]*2]); print('hold and latent drift loadings',np.linalg.svd(np.linalg.lstsq(np.c_[o,H,X],V,rcond=None)[0][2:]@np.linalg.pinv(T.T),full_matrices=False)[1])
base={'joint_diag':X,'joint_all':Xall,'rawzero':raw0,'rawfill':np.where(imiss,np.nanmean(Xraw,0),Xraw),'diag_only':m._features(c,c)[:len(lab)]}
for swap in ['reference','coupon','raw']:
 for pool in ['train','diag','all','train_test']:
  ll=np.concatenate([q['lab_id'] for q in {'train':[a],'diag':[c],'all':[a,b,c],'train_test':[a,b]}[pool]]); xx=np.vstack([q['ref_coupons' if swap=='reference' else 'coupon_features_raw'] for q in {'train':[a],'diag':[c],'all':[a,b,c],'train_test':[a,b]}[pool]]); F=Xraw.copy()
  for l in np.unique(lab):
   p=xx[ll==l]; cen=np.nanmedian(np.nanmean(p,axis=0)); sd=np.nanmedian(np.nanstd(p,axis=0)); F[lab==l]=(F[lab==l]-cen)/(sd if swap!='raw' else 1.)
  base[swap+'_'+pool]=np.where(imiss,np.nanmean(F,0),F)
for name,F in base.items():
 for sub in ['drift','features','drift_lab']:
  dr=th[2]+F@th[3:]; A=np.c_[o,H,dr] if sub=='drift' else np.c_[o,H,F] if sub=='features' else np.c_[o,H,dr,dr*(M[lab,0]-1),M[lab,0]-1]; AT=np.linalg.lstsq(A,V,rcond=None)[0]; Q=AT@np.linalg.pinv(T.T); pred=A@Q@T.T; ee=[]
  for l in np.unique(lab):
   z=lab!=l; ee.append(A[~z]@np.linalg.lstsq(A[z],V[z]@np.linalg.pinv(T.T),rcond=None)[0]@T.T-V[~z])
  def sc(E):return np.array([np.mean(E[:,:-1]**2)**.5,np.mean(E[:,-1]**2)**.5/2])
  print(name,sub,sc(pred-V).round(5),np.max([sc(e) for e in ee],0).round(5))
  if name=='joint_diag' and sub=='drift':
   u,s,v=np.linalg.svd(V-pred,full_matrices=False); print('row residual scores',np.c_[lab,imiss.sum(1),u[:,0]*s[0],Xraw].round(4)); print('residual temporal basis',s[:5],v[0].round(3))
