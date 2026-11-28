from p import *
idx=[c for k in range(4) for c in combinations_with_replacement(range(4),k)]
A=poly(3)
P=np.linalg.lstsq(A,Y,rcond=None)[0]
Q=np.linalg.lstsq(A[:52],Y[:52],rcond=None)[0]
print('monomial: response full/prefix, trajectory first/last full/prefix')
for c,p,q in zip(idx,P,Q): print(''.join(map(str,c)) or '1',np.round([p[0],q[0],p[1],q[1],p[-1],q[-1]],4))
print('trajectory singular values',np.linalg.svd(T,compute_uv=False).round(4))
print('coefficient singular values',np.linalg.svd(P[:,1:],compute_uv=False).round(4))
def fit(ii,j,cut):
 a=A[ii]; b=Y[ii,j]; keep=np.arange(a.shape[1]); sc=np.sqrt((a*a).mean(0)); a=a/sc
 while True:
  co=np.linalg.lstsq(a[:,keep],b,rcond=None)[0]; valid=np.max(abs(co),axis=1)>cut if co.ndim==2 else abs(co)>cut
  if np.all(valid):break
  keep=keep[valid]
 return keep,co/sc[keep,None] if co.ndim==2 else co/sc[keep]
for j in (0,slice(1,None)):
 for cut in (.006,.012,.02,.03,.045,.06,.08):
  k,c=fit(np.arange(52),j,cut); ee=[]
  for ii in np.array_split(np.random.default_rng(39).permutation(72),6):
   kk,cc=fit(np.setdiff1d(np.arange(72),ii),j,cut); ee.append(np.mean(abs(Y[ii,j]-A[ii][:,kk]@cc)))
  print(j,cut,len(k),'tail',round(np.mean(abs(Y[52:,j]-A[52:][:,k]@c)),5),'folds',np.round(ee,4),'terms',[idx[i] for i in k])
