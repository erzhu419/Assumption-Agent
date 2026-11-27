import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): v=runpy.run_path('/tmp/chamber_probe.py')
g,tr,te,t,dt,a,h,b,Y,p,l=(v[k] for k in ('g','tr','te','t','dt','a','h','b','Y','p','l'))
s=v['s']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def align(parts):
 z={k:np.concatenate([q[k] for q in parts]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; lab=z['lab_id']; X=z['coupon_features_raw']; _,ix,n=np.unique(np.c_[lab,z['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[ix];
 for j in np.unique(lab):
  sel=lab==j; R=z['ref_coupons'][sel]; X[sel]=(X[sel]-np.median(R.mean(0)))/np.median(R.std(0))
 for c in X.T:
  ok=np.isfinite(c); c[~ok]=np.average(c[ok],weights=w[ok])
 return X
G=np.c_[np.ones(len(l)),l,p]
def gap(taus):
 z=np.zeros((len(l),len(taus)))
 for width,ps in zip(dt,p.T):z+=-np.expm1(-width/taus)*(ps[:,None]-z)
 return l[:,None]-z

def score(D):
 coeff=np.linalg.lstsq(D,Y,rcond=None)[0]; E=Y-D@coeff; loo=E/(1-np.sum(np.linalg.svd(D,full_matrices=False)[0]**2,axis=1))[:,None]; return np.sqrt(np.mean(E**2)), np.sqrt(np.mean(loo**2)), np.max(abs(loo)), coeff
for tag,X in [('train_diag',v['X']),('diag',align([g])),('joint',align([tr,te,g])[-len(l):]),('train_test',None)]:
 if X is None:
  D={k:np.concatenate([q[k] for q in (tr,te)]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; Z=align([tr,te]); X=g['coupon_features_raw'].copy(); _,ix,n=np.unique(np.c_[D['lab_id'],D['stratum']],axis=0,return_inverse=True,return_counts=True)
  for j in np.unique(g['lab_id']):
   R=D['ref_coupons'][D['lab_id']==j]; sel=g['lab_id']==j; X[sel]=(X[sel]-np.median(R.mean(0)))/np.median(R.std(0))
  for k,c in enumerate(X.T):
   ok=np.isfinite(D['coupon_features_raw'][:,k]); c[~np.isfinite(c)]=np.average(Z[ok,k],weights=1/n[ix[ok]])
 r=h[2]+X@h[3:]; H=gap(a); print('\n',tag)
 designs={'all-linear':np.c_[G,X], 'memory-drift':np.c_[np.ones(len(l)),l,H,r], 'memory-features':np.c_[np.ones(len(l)),l,H,X], 'stress-drift':np.c_[G,r], 'stress-drift-interact':np.c_[G,r,l*r,r*r], 'memory-smooth':np.c_[np.ones(len(l)),l,H,r,r*r,l*r,np.tanh(r),l*l]}
 for name,D in designs.items():
  e,f,m,C=score(D); print(name, np.round([e,f,m],6))
  if name in ('all-linear','memory-drift','stress-drift'):print('coeff early/last/rate',C[:,[0,-2,-1]].T,'singular',np.linalg.svd(C,compute_uv=False))
 if tag in ('joint','train_test'):
  E=Y-designs['memory-drift']@score(designs['memory-drift'])[-1]; U,S,V=np.linalg.svd(E,full_matrices=False); print('memory residual row rms',np.sqrt(np.mean(E**2,axis=1))); print('resid singular',S,'mode',V[:2]); print('r',r)
  np.savez('/tmp/chamber_'+tag+'.npz',Y=Y,t=t,dt=dt,a=a,h=h,b=b,p=p,l=l,X=X,r=r,lab=g['lab_id'],H=H)
