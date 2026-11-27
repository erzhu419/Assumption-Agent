exec(open('/tmp/m99_chamber.py').read().split("for name,A,V in")[0])
_, xjoint=m._features(tr,{k:np.concatenate([te[k],g[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}); dj=beta[2]+xjoint[len(xt):]@beta[3:]
D=C-reset; print('reset correction singular values',np.linalg.svd(D,compute_uv=False)); print('correction on physical observed scales',np.linalg.svd(D@F.T,compute_uv=False)[:5]); print('lab, missing, drift, joint drift, hold, memory, coefficient shifts, onset shift'); print(np.c_[g['lab_id'],np.isnan(g['coupon_features_raw']).sum(1),drift,dj,hg,z,D,D@F[-1]])
from itertools import combinations_with_replacement
n=len(hg); inds=np.arange(n); one=np.ones(n)
sets={'drift': [drift], 'drift+hold':[drift,hg], 'drift+memory':[drift,*z.T], 'drift+hold+memory':[drift,hg,*z.T], 'features':list(xg.T)}
for k,cols in sets.items():
 for deg in (1,2,3):
  if k=='features' and deg==3: continue
  A=np.column_stack([one]+[np.prod(seq,axis=0) for pwr in range(1,deg+1) for seq in combinations_with_replacement(cols,pwr)])
  if A.shape[1]>=n-4: continue
  rs=[]
  for i in inds:
   mask=inds!=i
   rs.append(A[i]@np.linalg.lstsq(A[mask],D[mask],rcond=None)[0]@F.T+reset[i]@F.T-Y[i])
  E=np.array(rs); print(k,deg,np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),np.max(abs(E)))
