from chamber_cv import *
np.set_printoptions(precision=5,suppress=True)
z=feat('self'); d=beta[2]+z@beta[3:]; h=hq; o=np.ones(len(h)); g=prior._gap(tau,pq,h,dt); M=np.isnan(q['coupon_features_raw']).any(1)
print('range drift and hold',d.min(),d.max(),h.min(),h.max())
# Use the exact fast hold amplitude implied by the intrinsic material, and test
# additional terms against independent lab folds rather than fitting noisy free amplitudes.
for ratio in [1.,1.15,1.18,1.2]:
 ts=tau*np.array([1,ratio]); K=modes(ts); base=h[:,None]*beta[0]*K[None,:,0]; yy=Y-base
 for name,b in [('core',[np.c_[d],np.c_[h,d],np.c_[d]]),('affine',[np.c_[o,d],np.c_[o,h,d],np.c_[o,d]]),('interaction',[np.c_[o,d],np.c_[o,h,d,h*d],np.c_[o,d]]),('drift2',[np.c_[o,d,d*d],np.c_[o,h,d,d*d],np.c_[o,d,d*d]]),('memory',[np.c_[o,d],np.c_[o,h,d,g],np.c_[o,d]])]:
  A=np.concatenate([v[:,None,:]*K[None,:,j,None] for j,v in enumerate(b)],axis=2); n=A.shape[-1]
  def solve(rows):
   a=A[rows].reshape(-1,n); y=yy[rows].ravel(); c=np.linalg.lstsq(a,y,rcond=None)[0]; return A@c+base,c
  pred=Y.copy(); worst=[]
  for lab in np.unique(q['lab_id']):
   v=q['lab_id']==lab; p,_=solve(~v); pred[v]=p[v]; e=p[v]-Y[v]; worst.append([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
  p,c=solve(np.ones(len(Y),bool)); e=pred-Y; print(ratio,name,np.round(np.max(worst,axis=0),5),'complete/missing rms',np.round([np.sqrt(np.mean(e[~M]**2)),np.sqrt(np.mean(e[M]**2))],5),'coef',c)
  if ratio==1.18 and name=='affine':
   print('row residuals: lab hold drift miss curve_end rate'); print(np.c_[q['lab_id'],h,d,M,(p-Y)[:,-2:]])
