from m99_probe import *
from itertools import product
xs,_=m._features(c,c); drift=beta[2]+xs@beta[3:]; labs=c['lab_id']; complete=np.isfinite(c['coupon_features_raw']).all(1)
P=ac[:,:,:2]@beta[:2]; R=bc[:,:2]@beta[:2]; E=-np.expm1(-t[:,None]/tau); U=E*tau-t[:,None]; q=drift[:,None,None]*U; target=yc-P-drift[:,None]*t
baseline=P+drift[:,None]*t-yc
for f in [(0.7,0.55),(0.68,0.58),(0.68,0.57),(0.69,0.56)]:
 e=q@f-target; print(f,'RMSE complete/incomplete',*[np.sqrt(np.mean(e[s]**2)) for s in (complete,~complete)],'signed residual by horizon',np.round(e[complete].mean(0),5))
for label,g in [('lab',labs),('stratum',c['stratum']),('signed drift',drift<0),('hold tertile',np.digitize(hc,np.quantile(hc,[1/3,2/3])))]:
 e=np.zeros_like(yc); a=[]
 for l in np.unique(g):
  test=g==l; fit=complete&~test; f=np.linalg.lstsq(q[fit].reshape(-1,2),target[fit].ravel(),rcond=None)[0]; e[test]=q[test]@f-target[test]; a.append((l,np.round(f,4),round(np.sqrt(np.mean(e[test]**2)),5),round(np.sqrt(np.mean((R+drift-rc)[test]**2)),5),round(np.sqrt(np.mean(baseline[test]**2)),5)))
 print(label,'[fold, fractions, strain RMSE, rate RMSE, baseline strain RMSE]',a)
print('fit constrained with steady fraction -.25',np.linalg.lstsq((q[:,:,0]-q[:,:,1])[complete].reshape(-1,1),(target-1.25*q[:,:,1])[complete].ravel(),rcond=None)[0]); print('temporal fit covariance collinearity',np.linalg.svd(q[complete].reshape(-1,2),compute_uv=False)); print('dt',dt,'times',t)
