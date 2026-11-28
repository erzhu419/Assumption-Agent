import pickle,numpy as np
from scipy.optimize import minimize_scalar
from m100_base import _bridge,_loads,_features,_design,_fit
np.set_printoptions(precision=5,suppress=True,linewidth=170)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; q=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; n=len(tr['lab_id']); x=_features(tr,te); xd=_features(dict((k,np.concatenate((tr[k],te[k]))) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')),q)[-len(q['lab_id']):]; b=_bridge(d['calibration_load_cells']); tau,theta=_fit(*_loads(tr,b),x[:n],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); p,h=_loads(q,b); C,R=_design(tau,p,h,xd,dt,t); ec=q['strain_curves_obs']-q['strain_0'][:,None]-C@theta; er=q['instant_rate_obs']-R@theta; E=np.c_[er,ec]
def temporal(k,rate=True):
 L=np.c_[-np.expm1(-t/k),t]; return np.vstack(([1/k,1],L)) if rate else L
def err(k,rate=True):
 L=temporal(k,rate); Y=E if rate else ec; return Y-(Y@np.linalg.pinv(L.T))@L.T
from scipy.optimize import least_squares
D=np.c_[q['instant_rate_obs'],q['strain_curves_obs']-q['strain_0'][:,None]]
T=np.r_[1.,t]; L=lambda k:np.r_[1/k,-np.expm1(-t/k)]
for k in ([.58,3.1],[.58,4.2],[.58,4.35]):
 K=np.array([L(j) for j in k]+[T]); A=D@np.linalg.pinv(K); print('taus and amplitude relation',k,np.linalg.lstsq(np.c_[np.ones(len(h)),h,A[:,0]],A[:,1:],rcond=None)[0]); print('amplitude residual RMS',np.sqrt(np.mean((A[:,1:]-np.c_[np.ones(len(h)),h,A[:,0]]@np.linalg.lstsq(np.c_[np.ones(len(h)),h,A[:,0]],A[:,1:],rcond=None)[0])**2,axis=0)))
K=np.array([L(.58),L(3.1),T]); A=D@np.linalg.pinv(K); f=A[:,0]-.81*h; raw=q['coupon_features_raw']; lab=q['lab_id']; miss=np.isnan(raw).any(1); u=np.nan_to_num(raw)@theta[3:]; M=np.c_[np.eye(4)[lab],np.eye(4)[lab]*u[:,None]]; co=np.linalg.lstsq(M[~miss],f[~miss],rcond=None)[0]; print('raw scalar lab intercept/scale',co, 'complete RMSE',np.mean((f-M@co)[~miss]**2)**.5); print('lab, missing, projected amplitude, scalar frame, calibrated residual'); print(np.c_[lab,miss,f,xd@theta[3:],f-M@co].round(5)); print('reference stats mean/std');
for label,part in [('tr',tr),('te',te),('q',q)]:
 print(label,[(np.median(part['ref_coupons'][part['lab_id']==j].mean(0)),np.median(part['ref_coupons'][part['lab_id']==j].std(0))) for j in range(4)])
for nm,X in [('lab-affine complete',M),('train+q drift',np.c_[np.ones(len(h)),_features(tr,q)[-len(h):]@theta[3:]]),('pooled drift',np.c_[np.ones(len(h)),xd@theta[3:]])]:
 for extra in ['none','linear','terminal','quadratic']:
  V={'none':np.empty((len(h),0)),'linear':p,'terminal':p[:,-2:],'quadratic':np.c_[p[:,-2:],p[:,-2]*p[:,-1],p[:,-2:]**2]}[extra]; F=np.c_[X,h,V]; keep=~miss if nm.startswith('lab-affine') else np.ones(len(h),bool); P=F@np.linalg.lstsq(F[keep],D[keep],rcond=None)[0]; print(nm,extra,np.sqrt(np.mean((P-D)[keep]**2)),np.max(abs((P-D)[keep])))
print('test hold/history ranges',[(a.min(),a.max()) for a in _loads(te,b)])
