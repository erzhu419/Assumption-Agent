import pickle,numpy as np
from scipy.optimize import least_squares
import w04_prior as m
D=pickle.load(open('/holdout_task/public_alpha_w04_batch_07.pkl','rb'))[0]['dataset']; tr,te=D['train'],D['test']; dt,t=D['past_dt'],D['eval_times']; x,_=m._align(tr,te); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; lab=tr['lab_id']; n,k=y.shape; complete=np.isfinite(tr['coupon_features_raw']).all(axis=1); base=m._calibrate(D['calibration_load_cells']); Y=D['yield_stress'];
def state(a): return m._prepare(dt,*m._stress(tr,a),x,Y)
def report(label,a,ta,w):
 st=state(a); p,q=m._design(ta,t,st); e=p@w-y; er=q@w-r; print(label,'affine',np.round(a[2:],5).tolist(),'taus',np.round(ta,5),'weights',np.round(w,5),'plastic',*[np.count_nonzero(z) for z in st[-2:]],flush=True)
 for j in range(4):
  z=(lab==j)&complete; print(j,'strain med abs/rmse',np.median(abs(e[z])),np.sqrt(np.mean(e[z]**2)),'rate med abs/rmse',np.median(abs(er[z])),np.sqrt(np.mean(er[z]**2)),'mean curve',np.round(e[z].mean(0)[[0,4,9]],4),flush=True)
a=base.copy(); ta,w=m._fit(t,state(a),y,r); report('archive',a,ta,w)
for j in range(2,4):
 u,v=m._rows(D['calibration_load_cells'],j); a[j]=m._line(u[:2],v[:2]);
tc,wc=m._fit(t,state(a),y,r); report('first pair',a,tc,wc)
# Estimate Wing B stress from coupon response with Wing A calibrated; use complete rows to isolate calibration error.
def residual(q,plastic):
 af=base.copy(); af[2:]=q[9:13].reshape(2,2); P,R=m._design(q[:2],t,state(af)); W=np.r_[q[2:9],q[13:15]] if plastic else np.r_[q[2:9],0.,0.]; return np.r_[(P[complete]@W-y[complete]).ravel(),(R[complete]@W-r[complete])*2]
for af,ts,ws in ((base,ta,w),(a,tc,wc)):
 for plastic in (False,True):
  f=least_squares(lambda q:residual(q,plastic),np.r_[ts,ws[:7],af[2:].ravel(),ws[7:]] if plastic else np.r_[ts,ws[:7],af[2:].ravel()],loss='soft_l1',f_scale=.02,x_scale='jac',max_nfev=120,bounds=(np.r_[.12,1.,np.full(7,-np.inf),.15,-8,.15,-8,([-np.inf]*2 if plastic else [])],np.r_[1.5,8.,np.full(7,np.inf),3,8,3,8,([np.inf]*2 if plastic else [])])); q=f.x; b=base.copy(); b[2:]=q[9:13].reshape(2,2); report('joint '+str(plastic)+' cost '+str(round(f.cost,6)),b,q[:2],np.r_[q[2:9],q[13:15]] if plastic else np.r_[q[2:9],0.,0.]);
  for j in (2,3):
   u,v=m._rows(D['calibration_load_cells'],j); print('check residual',j,np.round(v-(b[j,0]*u+b[j,1]),5));
