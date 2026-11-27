from constit06 import *
from scipy.linalg import lstsq
lab=g['lab_id']; labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); raw=g['coupon_features_raw'].copy();
for j in np.unique(lab):
 a=refs[labs==j]; raw[lab==j]=(raw[lab==j]-np.median(np.nanmean(a,0)))/np.median(np.nanstd(a,0))
print('diag labs and missing',lab,np.isnan(raw).astype(int)); print('feature change',np.round(raw-xg[:,1:],3));
lst=np.r_[tr['stratum'],te['stratum']]; _,cell,cnt=np.unique(np.c_[labs,lst],axis=0,return_inverse=True,return_counts=True); w=1/cnt[cell]; mat=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); mu=np.array([np.average(xx[~np.isnan(col),j+1],weights=w[~np.isnan(col)]) for j,col in enumerate(mat.T)]); x=np.c_[np.ones(len(lab)),np.where(np.isnan(raw),mu,raw)]; dr=x@theta[2:]; print('obs rate vs new reset',np.round(r-base@(1/tau)-dr,4)); print('drifts',np.round(np.c_[dr,xg@theta[2:],r-base@(1/tau),b[:,2]],4)); print('times',t)
# Each column is a distinct observable mechanism, observed both in curves and at the initial derivative.
T=np.c_[F,t]; deriv=np.r_[1/tau,1]; base_y=base@F.T; base_r=base@(1/tau); target=np.c_[y-base_y,r-base_r];
def mk(q,j): return q[:,None]*np.r_[T[:,j],deriv[j]]
def evaluate(name,terms,offset=0):
 A=np.stack(terms,axis=-1); off=np.broadcast_to(offset,target.shape); wt=np.r_[np.ones(len(t)),2.]; E=np.empty_like(target)
 for j in range(len(x)):
  ii=np.arange(len(x))!=j; c=lstsq((A[ii]*wt[None,:,None]).reshape(-1,A.shape[2]),((target-off)[ii]*wt).ravel())[0]; E[j]=(target-off)[j]-A[j]@c
 c=lstsq((A*wt[None,:,None]).reshape(-1,A.shape[2]),((target-off)*wt).ravel())[0]; print(name,'CV rms',np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2)),'worst coupon',max(np.sqrt(np.mean(E[:,:-1]**2,1))),max(abs(E[:,-1])),'coef',np.round(c,5)); return c
for label, ddr in [('joint',dr),('diag',xg@theta[2:])]:
 drift=[mk(ddr*tau[j],j) for j in range(2)]+[mk(ddr,2)]; history=[mk(Z[:,j]*theta[j],j) for j in range(2)]; holds=[mk(base[:,j],j) for j in range(2)];
 evaluate(label+' dr',drift); evaluate(label+' history dr',history+drift); evaluate(label+' stress dr',holds+history+drift); evaluate(label+' full features drift gauge',[sum(mk(x[:,k]*tau[j]*a,j) for j,a in enumerate([.52,.43]))+.05*mk(x[:,k],2) for k in range(5)]); evaluate(label+' history full features drift',history+[mk(x[:,k],j) for j in range(3) for k in range(5)])
