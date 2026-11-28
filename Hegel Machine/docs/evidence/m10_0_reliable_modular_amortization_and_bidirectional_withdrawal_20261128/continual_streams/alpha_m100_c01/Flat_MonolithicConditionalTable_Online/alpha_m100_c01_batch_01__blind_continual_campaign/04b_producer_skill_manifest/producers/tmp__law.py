from explore import *
from scipy.optimize import least_squares
p,h,F=inputs[2]; Y=di['strain_curves_obs']-di['strain_0'][:,None]; R=di['instant_rate_obs']; yy=np.c_[Y,R]; n=len(h); labs=di['lab_id']; ones=np.ones(n)
def basis(a,b): return np.vstack([np.c_[-np.expm1(-t[:,None]/[a,b]),t],np.r_[1/np.array([a,b]),1]])
D=basis(.497,3.356); gain=np.linalg.lstsq(D[:,:2],yy.T,rcond=None)[0].T
xdi=m._features(di,di)[0]; zs={'pooled':F[:,1:],'diag':xdi}
# Also transform diagnostics with train+test panel and imputation moments, without augmenting it.
raw=di['coupon_features_raw'].copy(); L=np.r_[tr['lab_id'],te['lab_id']]; ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); X=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]);
for lab in np.unique(L):
 ref0=ref[L==lab]; c=np.median(ref0.mean(0)); s=np.median(ref0.std(0)); raw[labs==lab]=(raw[labs==lab]-c)/s; X[L==lab]=(X[L==lab]-c)/s
_,cells,counts=np.unique(np.c_[L,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); w=1/counts[cells,None]; mu=np.nansum(w*X,axis=0)/(w*np.isfinite(X)).sum(0); zs['train+test']=np.where(np.isnan(raw),mu,raw)
for key,X0 in zs.items():
 r=theta[2]+X0@theta[3:]; print(key,'r, amp0 - theta0*hold, amp1 - theta1*hold, amp1/hold',np.round(np.c_[r,gain[:,0]-theta[0]*h,gain[:,1]-theta[1]*h,gain[:,1]/h],4) if key=='pooled' else '')
 out=[]
 for typ in ['r','r2','r3','x','h*r','h*r2','h*x','past+r','past+h*r']:
  f={'r':np.c_[h,ones,r],'r2':np.c_[h,ones,r,r*r],'r3':np.c_[h,ones,r,r*r,r**3],'x':np.c_[h,ones,X0],'h*r':np.c_[h,h*r],'h*r2':np.c_[h,h*r,h*r*r],'h*x':np.c_[h,h[:,None]*X0],'past+r':np.c_[h,p,ones,r],'past+h*r':np.c_[h,p,h*r]}[typ]
  pred=np.empty_like(yy)
  for lab in np.unique(labs):
   keep=labs!=lab; pred[~keep]=np.c_[gain[~keep,0],f[~keep]@np.linalg.lstsq(f[keep],gain[keep,1],rcond=None)[0]]@D[:,:2].T
  out.append((typ,round(rms(pred[:,:-1]-Y),5),round(rms(pred[:,-1]-R),5),np.round(np.linalg.lstsq(f,gain[:,1],rcond=None)[0],4)))
 for v in out: print(v)
print('missing by diag feature',np.isnan(di['coupon_features_raw']).sum(0)); print('corr f0 hold',np.corrcoef(h,gain[:,0]), 'amp0 line', np.linalg.lstsq(np.c_[ones,h],gain[:,0],rcond=None)[0]); print('gain0 residual vs gain1',np.linalg.lstsq(np.c_[ones,h,gain[:,1]],gain[:,0],rcond=None)[0])
