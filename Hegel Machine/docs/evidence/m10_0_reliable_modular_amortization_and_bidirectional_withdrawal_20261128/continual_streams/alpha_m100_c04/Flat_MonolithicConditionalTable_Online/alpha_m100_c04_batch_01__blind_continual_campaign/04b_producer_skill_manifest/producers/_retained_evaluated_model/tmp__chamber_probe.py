import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m100_c04_s00_solver as s
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; q=item['campaign_diagnostics']; tr=d['train']; te=d['test']; t=d['eval_times']; dt=d['past_dt']; n=len(tr['lab_id']); c=s._calibration(d['calibration_load_cells']); xx=s._features(tr,te); out=s.solve_batch(d); tau=out['estimated_taus']; w=out['hereditary_weights']; print('material',tau,w)
# Apply the deployment panel to diagnostic features; diagnosis does not change material fit.
ref=np.vstack((tr['ref_coupons'],te['ref_coupons'])); labs=np.r_[tr['lab_id'],te['lab_id']]; x=q['coupon_features_raw'].copy(); cells=np.c_[labs,np.r_[tr['stratum'],te['stratum']]]; _,inv,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True); wt=1/counts[inv,None]; raw=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])); mean=(wt*xx*np.isfinite(raw)).sum(0)/(wt*np.isfinite(raw)).sum(0)
for lab in np.unique(labs):
 p=ref[labs==lab]; m=q['lab_id']==lab; x[m]=(x[m]-np.median(p.mean(0)))/np.median(p.std(0))
x=np.where(np.isfinite(x),x,mean); p,h,v=s._inputs(q,x,c); A,R=s._design(tau,(p,h,v),dt,t); y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; Y=y-np.c_[A@w,R@w]; print('baseline rms',np.sqrt(np.mean(Y**2,axis=0)).round(4))
def basis(z):
 return np.r_[np.c_[-np.expm1(-t[:,None]/z),t],[np.r_[1/z,1]]]
def coef(z):
 B=basis(z); return np.linalg.lstsq(B,y.T,rcond=None)[0].T
z=least_squares(lambda z:(coef(z)@basis(z).T-y).ravel(),tau,bounds=(.06,24)).x; print('shared exploratory chamber tau',z,'rms fixed/free',*[np.sqrt(np.mean((coef(j)@basis(j).T-y)**2,axis=0)).round(5) for j in (tau,z)])
# Regress each component of the discrepancy on physically meaningful inputs.
B=basis(tau); a=np.linalg.lstsq(B,Y.T,rcond=None)[0].T
P=np.c_[h,p]; gap=R[:,:2]*tau; drift=v@w[2:]; sums=np.c_[p.sum(1),p[:,:3].sum(1),p[:,3:].sum(1),p[:,::2].sum(1),p[:,1::2].sum(1),np.diff(p,axis=1).sum(1)]; F=np.c_[P,gap,drift,x,sums,q['stratum'],q['lab_id']]; names='hold p0 p1 p2 p3 p4 p5 gap0 gap1 drift x0 x1 x2 x3 psum first last even odd delta stratum lab'.split(); cor=np.corrcoef(np.c_[F,a].T)[:-3,-3:];
for name,row in zip(names,cor):print(name,row.round(3))
for name,J in [('past',p),('stress',P),('gap',gap),('drift',v[:,1:]),('stress+drift',np.c_[P,x]),('stress+drift+lab',np.c_[P,x,np.eye(4)[q['lab_id']][:,:3]])]:
 J=np.c_[np.ones(len(h)),J]; ae=np.zeros_like(a)
 for i in range(len(h)):
  m=np.arange(len(h))!=i; ae[i]=J[i]@np.linalg.lstsq(J[m],a[m],rcond=None)[0]
 err=(ae-a)@B.T; print(name,'coeffs',np.linalg.lstsq(J,a,rcond=None)[0].round(3).T,'LOO comp/curve/rate',np.sqrt(np.mean((ae-a)**2,axis=0)).round(4),np.sqrt(np.mean(err[:,:-1]**2)).round(4),np.sqrt(np.mean(err[:,-1]**2)).round(4))
