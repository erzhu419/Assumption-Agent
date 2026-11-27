from m99_inputs import *
D=np.vstack((E,er)); obs=np.c_[y,yr]; B=np.linalg.lstsq(D,obs.T,rcond=None)[0].T
p,h,x=cg; dr=x@th[2:]; z=h[:,None]-R0[:,:2]*tau; lab=g['lab_id']; s=g['stratum']; n=len(h)
print('diagnostic ranges from train/test reference scaling vs joint scaling')
for l in np.unique(lab):
 print(l,[(np.median(k['ref_coupons'][k['lab_id']==l].mean(0)),np.median(k['ref_coupons'][k['lab_id']==l].std(0))) for k in (tr,te,g)],'missing',np.isnan(g['coupon_features_raw'][lab==l]).sum())
print('corr B, nominal drift, hold, features, memory'); print(np.corrcoef(np.c_[B,dr,h,x[:,1:],z].T)[:3].round(3))
print('single channel sweep correlations'); sc=cal['single_channel_sweeps']; print({k:np.shape(v) for k,v in sc.items()}); print(np.c_[sc['lab_id'],sc['cmd_stress'],sc['true_stress']][:12].round(5))
models={'nominal':np.c_[h,z,dr], 'nominal+delta':np.c_[h,z,p[:,-1]-p[:,0],dr], 'hold+drift':np.c_[h,dr], 'hold+early/late+drift':np.c_[h,p[:,:3].mean(1),p[:,3:].mean(1),dr], 'history+drift':np.c_[h,p,dr], 'hold+memory+features':np.c_[h,z,x[:,1:]],'history+features':np.c_[h,p,x[:,1:]],'history+drift+nonlinear':np.c_[h,p,dr,dr**2,h*dr], 'hold+lab+stratum+drift':np.c_[h,lab,s,dr]}
for name,F in models.items():
 F=np.c_[np.ones(n),F]; co=np.linalg.lstsq(F,B,rcond=None)[0]; pred=np.zeros_like(B)
 for l in np.unique(lab):
  v=lab==l; pred[v]=F[v]@np.linalg.lstsq(F[~v],B[~v],rcond=None)[0]
 res=pred@D.T-obs; print(name,'fit coef rmse',np.mean((F@co-B)**2,0)**.5,'worst lab curve/rate RMSE',max(np.mean(res[lab==l,:-1]**2)**.5 for l in np.unique(lab)),max(np.mean(res[lab==l,-1]**2)**.5 for l in np.unique(lab))); print(co.round(4).T)
print('coef residual after hold+past+drift per sample'); F=np.c_[np.ones(n),h,p,dr]; print(np.c_[lab,s,np.isnan(g['coupon_features_raw']).sum(1),h,dr,B-F@np.linalg.lstsq(F,B,rcond=None)[0]].round(4))
