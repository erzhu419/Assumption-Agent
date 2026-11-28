exec(open('/tmp/probe_chamber.py').read().split('print(\'free amplitudes')[0])
from scipy.optimize import least_squares
L=di['lab_id']; C=np.column_stack((Y,R)); B=(h[:,None]*b[:2])@Q[:,:2].T; E=C-B

def feature_transform(panel, subset):
 labs=np.concatenate([s['lab_id'] for s in panel]); refs=np.vstack([s['ref_coupons'] for s in panel]); strata=np.concatenate([s['stratum'] for s in panel]); xx=np.vstack([s['coupon_features_raw'] for s in panel]); x=subset['coupon_features_raw'].copy()
 for l in np.unique(labs):
  ix=labs==l; P=refs[ix]; mu=np.median(P.mean(0)); sd=np.median(P.std(0)); xx[ix]=(xx[ix]-mu)/sd; j=subset['lab_id']==l; x[j]=(x[j]-mu)/sd
 _,i,n=np.unique(np.column_stack((labs,strata)),axis=0,return_inverse=True,return_counts=True); w=1/n[i,None]
 return np.where(np.isnan(x),np.nansum(xx*w,0)/np.sum(w*~np.isnan(xx),0),x)

def expk(T): return np.r_[-T*np.expm1(-t/T),1.]
def k(q):return q[1]*expk(q[0])+(1-q[1])*expk(q[2])
for name,panel in [('deploy',[tr,te]),('tr+di',[tr,di]),('di',[di]),('all',[tr,te,di]),('te+di',[te,di]),('train',[tr])]:
 x=feature_transform(panel,di); D=b[2]+x@b[3:]; missing=np.isnan(di['coupon_features_raw']).any(1); err=D-E[:,-1]; rr=np.sqrt(np.mean(err**2)); print(name,'rate overall, complete, missing, worst lab',np.round([rr,np.sqrt(np.mean(err[~missing]**2)),np.sqrt(np.mean(err[missing]**2)),max(np.sqrt(np.mean(err[L==l]**2)) for l in np.unique(L))],5))
 print(' residual by lab',[(int(l),np.round([err[L==l].mean(),np.std(err[L==l])],4).tolist()) for l in np.unique(L)])
 def fit(m):return least_squares(lambda q:(D[:,None]*k(q)-E)[m].ravel(),[1,.4,8],bounds=([.1,0,1],[5,1,100]),loss='soft_l1',f_scale=.007).x
 cvs=[(D[:,None]*k(fit(L!=l))-E)[L==l,:-1] for l in np.unique(L)];print(' kernel fit',np.round(fit(L>=0),4),'curve worst lab',max(np.sqrt(np.mean(c*c)) for c in cvs))
print('diagnostic missing rows',np.flatnonzero(missing))
# Fit the rank-one drift shape while allowing specimen drift amplitudes to vary, without using held-out curves to fit the shape.
for label in ['normalized exp+linear','normalized dual exp','normalized exp at intrinsic taus']:
 def kernel(q):
  if label.endswith('linear'):return q[1]*expk(q[0])+(1-q[1])*np.r_[t,1.]
  if label.endswith('dual exp'):return k(q)
  return q[0]*expk(tau[0])+(1-q[0])*expk(tau[1])
 ini,lo,hi=([1.4,.55],[.1,0],[12,1]) if label.endswith('linear') else (([1,.4,8],[.1,0,.2],[10,1,200]) if label.endswith('dual exp') else ([.3],[0],[1]))
 def fit(m):
  def resid(q):
   K=kernel(q); A=E[m]@K/(K@K); return (E[m]-A[:,None]*K).ravel()
  return least_squares(resid,ini,bounds=(lo,hi)).x
 q=fit(L>=0); K=kernel(q); D=E@K/(K@K); err=E-D[:,None]*K
 print(label,np.round(q,6),'projected residual RMSE',np.sqrt(np.mean(err**2)), 'kernel samples',np.round(K[[0,4,8,11,12]],5))
 for name,panel in [('deploy',[tr,te]),('tr+di',[tr,di]),('di',[di]),('all',[tr,te,di]),('te+di',[te,di])]:
  x=feature_transform(panel,di); P=b[2]+x@b[3:]; print(name,'drift RMSE',round(np.sqrt(np.mean((P-D)**2)),5),'complete',round(np.sqrt(np.mean((P[~missing]-D[~missing])**2)),5))
