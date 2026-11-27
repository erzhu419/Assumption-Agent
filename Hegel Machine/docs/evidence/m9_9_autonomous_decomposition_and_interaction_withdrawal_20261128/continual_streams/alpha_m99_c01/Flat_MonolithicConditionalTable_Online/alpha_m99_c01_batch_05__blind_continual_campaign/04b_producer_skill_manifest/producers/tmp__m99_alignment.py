exec(open('/tmp/m99_chamber.py').read().split('for name,A,V in')[0])
from itertools import combinations
np.set_printoptions(precision=5,suppress=True,linewidth=150)
H=np.c_[hg[:,None]*beta[:2],np.zeros(len(hg))]; Q=Y-H@F.T
u,s,v=np.linalg.svd(Q,full_matrices=False); print('hold-only residual singular values in measured units',s[:6]); print('leading temporal coefficient',np.linalg.lstsq(F,v[0],rcond=None)[0]); print('leading scores vs derived drift',np.c_[g['lab_id'],np.isnan(g['coupon_features_raw']).sum(1),drift,u[:,0]*s[0],Q[:,-1],g['ref_coupons'].mean(axis=1),g['ref_coupons'].std(axis=1)])
lab=g['lab_id']; raw=g['coupon_features_raw']; ids=np.unique(lab); L=(lab[:,None]==ids).astype(float); miss=np.isnan(raw).any(axis=1)
def stack(parts,k): return np.concatenate([p[k] for p in parts]);
def feats(parts,kind):
 x=raw.copy(); labs=stack(parts,'lab_id'); R=stack(parts,'ref_coupons');
 for ell in ids:
  r=R[labs==ell]; ind=lab==ell
  if kind=='column': mu, sd=np.median(r.mean(0)),np.median(r.std(0))
  elif kind=='row': mu, sd=np.median(r.mean(1)),np.median(r.std(1))
  elif kind=='pooled': mu, sd=r.mean(),r.std()
  elif kind=='median': mu, sd=np.median(r),1.4826*np.median(abs(r-np.median(r)))
  print('',end=''); x[ind]=(x[ind]-mu)/sd
 x=np.where(np.isnan(x),xg,x)
 return beta[2]+x@beta[3:]
def cv(A, base=H@F.T, select=np.ones(len(hg),bool)):
 yy=Y-base; e=[]
 for j in np.where(select)[0]:
  keep=select.copy(); keep[j]=False
  e.append(A[j]@np.linalg.lstsq(A[keep],yy[keep],rcond=None)[0]+base[j]-Y[j]);
 e=np.array(e); return np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.max(abs(e[:,:-1])),np.max(abs(e[:,-1]))
for names,parts in [('diag',[g]),('train',[tr]),('test',[te]),('train+diag',[tr,g]),('train+test',[tr,te]),('all',[tr,te,g])]:
 for kind in ('column','row','pooled','median'):
  dd=feats(parts,kind); print(names,kind,'global',np.round(cv(dd[:,None]),5),'lab',np.round(cv(np.c_[L,dd]),5),'complete',np.round(cv(dd[:,None],select=~miss),5))
for name,A in [('lab offsets',np.c_[L,drift]),('lab scale',L*drift[:,None]),('lab affine',np.c_[L,L*drift[:,None]]),('lab + features',np.c_[L,xg]),('load+lab affine',np.c_[hg,z,L,L*drift[:,None]]),('row refs',np.c_[drift,g['ref_coupons'].mean(1),g['ref_coupons'].std(1)])]:
 print(name,cv(A),'complete',cv(A,select=~miss)); W=np.linalg.lstsq(A,Q,rcond=None)[0]; print('   coef at onset and at end',W[:,[-1,-2]].T)
