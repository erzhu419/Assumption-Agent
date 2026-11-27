exec(open('/tmp/chamber_modes.py').read().split('for yname')[0])
X=m['_aligned_features'](q,q)[n:]; D=X@th[2:]; U=basis(tau); full=np.arange(n); G=A[:,0,:2]/(-np.expm1(-t[0]/tau)); z=b[:,None]-G
B=b[:,None]*(U[:,:2]@th[:2]); J=U.copy(); J[:,:2]*=tau; yd=Y-B; M=D[:,None,None]*J[None,:,:]; print('eval times',t,'drift range',D.min(),D.max()); print('diag keys',list(q)); print('test keys',list(te))
for name,base,S in [('free fractions',B,M),('conserved fractions',B+M[:,:,2],M[:,:,:2]-M[:,:,2:]),('conserved plus memory',B+M[:,:,2],np.dstack((M[:,:,:2]-M[:,:,2:],-z[:,None,:]*U[None,:,:2]*th[:2]))),('conserved plus hold scales',B+M[:,:,2],np.dstack((M[:,:,:2]-M[:,:,2:],b[:,None,None]*U[None,:,:2]*th[:2]))),('conserved plus offset',B+M[:,:,2],np.dstack((M[:,:,:2]-M[:,:,2:],J[None,:,:]*np.ones((n,1,1)))))]:
 k=S.shape[-1]
 def fit(keep): return np.linalg.lstsq(S[keep].reshape(-1,k),(Y-base)[keep].ravel(),rcond=None)[0]
 c=fit(full); eo=[]
 for labels in (full,q['lab_id'],q['stratum']):
  pred=np.empty_like(Y)
  for j in np.unique(labels):
   keep=labels!=j; pred[~keep]=base[~keep]+S[~keep]@fit(keep)
  e=abs(Y-pred); eo.append(np.round([np.quantile(e[:,:-1],.9),np.max(e[:,:-1].mean(1)),np.quantile(e[:,-1],.9),e[:,-1].max()],5))
 print(name,'coef',c.round(6),'CV coupon/lab/stratum (p90 curve, worst MAE, p90 rate, max rate)',eo)
print('conservation check, median/90th absolute initial-rate error',np.quantile(abs(yd[:,-1]-D),[.5,.9]))
