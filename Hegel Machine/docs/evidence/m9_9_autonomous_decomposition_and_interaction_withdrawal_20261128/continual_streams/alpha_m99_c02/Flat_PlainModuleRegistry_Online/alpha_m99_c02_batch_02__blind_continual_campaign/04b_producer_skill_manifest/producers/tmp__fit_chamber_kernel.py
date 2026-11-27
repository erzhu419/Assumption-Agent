import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_probe import *
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
Z=np.c_[y,0.6*q]; relax=-np.expm1(-t[:,None]/taus); E=np.vstack((relax,0.6/taus)); gap=r[:,:2]*taus; hist=h[:,None]-gap
D={}
for tag,sets in [('train+diag',(tr,di)),('train+test',(tr,te)),('all',(tr,te,di)),('diag',(di,)),('train',(tr,))]:
 x=di['coupon_features_raw'].copy()
 for lab in np.unique(di['lab_id']):
  panel=np.concatenate([s['ref_coupons'][s['lab_id']==lab] for s in sets]); m=di['lab_id']==lab
  x[m]=(x[m]-np.median(panel.mean(0)))/np.median(panel.std(0))
 # Compare normalization on complete rows so that imputation does not bias this comparison.
 D[tag]=theta[2]+x@theta[3:]
for fixed in [None,0.,1.]:
 for mode in ['fast+slow','fast+linear','single','twofree']:
  def prof(a, details=False):
   f,s,t0=a[:3]; mem=a[3:5] if fixed is None else fixed
   A=((h[:,None]-hist*mem)*theta[:2])@E.T
   fast = taus[0] if mode!='twofree' else t0
   Kfast=-fast*np.expm1(-t/fast)
   Kslow=-s*np.expm1(-t/s) if mode!='fast+linear' else t
   K=(f*Kfast+(1-f)*Kslow) if mode!='single' else Kslow
   K=np.r_[K,.6]; dr=(Z-A)@K/(K@K); err=A+dr[:,None]*K-Z
   return (dr,err,K) if details else err.ravel()
  fit=least_squares(prof,[.8,5.,.65,1.,1.],bounds=([0,.08,.08,-.5,-.5],[1,250,12,2,2]),loss='soft_l1',f_scale=.003,max_nfev=400)
  dr,err,K=prof(fit.x,True)
  print('memory',fixed,'mode',mode,'fit',fit.x,'kernel_end',K[-2],'rmse',rms(err),'max',np.max(abs(err)))
  if mode=='fast+slow' and fixed in [None,1.]:
   print('drift RMSE by panel, complete rows',[(k,rms((v-dr)[np.isfinite(v)])) for k,v in D.items()]); print('drift residuals',np.c_[di['lab_id'],dr,dr-D['train+diag'],dr-D['diag']].T)
