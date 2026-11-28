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
X=_features(q,q)[:len(h)]; s=np.c_[np.ones(len(h)),X]@theta[2:]; F=np.c_[np.ones(len(h)),s]; T=np.r_[1.,t]; L=lambda k:np.r_[1/k,-np.expm1(-t/k)]; K=lambda k:np.array([L(j) for j in k]); valid=~np.isnan(q['coupon_features_raw']).any(1)
for mode in ['free profiles','fixed hold','adjust hold','rankone','rankone exact','rankone no linear']:
 def fit(mask):
  if mode in ['free profiles','fixed hold']:
   G=np.c_[h,F] if mode=='free profiles' else F
   H=0 if mode=='free profiles' else h[:,None]*(theta[:2]@K(tau)); co=np.linalg.lstsq(G[mask],(D-H)[mask],rcond=None)[0]; return H+G@co,co
  def predict(k):
   if mode=='adjust hold':
    H=h[:,None]*(k[2:4]@K(k[:2])); A=np.vstack([K(k[:2]),T]); return H+F@np.linalg.lstsq(F[mask],(D-H)[mask],rcond=None)[0]@np.linalg.pinv(A)@A
   A=K(k[:2]); A=np.vstack([A,T]) if mode!='rankone no linear' else A
   H=h[:,None]*(k[2:4]@K(k[:2])); v=s+k[4] if mode!='rankone exact' else s
   z=(v[mask]@(D-H)[mask])/(v[mask]@v[mask]); return H+v[:,None]*(z@np.linalg.pinv(A)@A)
  k=least_squares(lambda k:(predict(k)-D)[mask].ravel(),[.58,4.2,.81,.57,-.09],bounds=([.3,1.5,.5,.2,-.5],[1.,12.,1.1,1.,.5]),max_nfev=150).x
  return predict(k),k
 P,B=fit(np.ones(len(h),bool)); E=P-D; ev=lambda z:np.sqrt(np.mean(z**2,axis=0)); print(mode,'params',B,'rmse complete/missing',ev(E[valid])[[0,-1]],ev(E[~valid])[[0,-1]])
 print('worst lab',np.max([ev((fit(q['lab_id']!=j)[0]-D)[q['lab_id']==j]) for j in np.unique(q['lab_id'])],axis=0)); print('missing residual',E[~valid][:,[0,-1]])
H=h[:,None]*(theta[:2]@K(tau)); B=np.linalg.lstsq(F[valid],(D-H)[valid],rcond=None)[0]; W=D-H-B[0]; inferred=(W@B[1])/(B[1]@B[1]); ii,jj=np.nonzero(np.isnan(q['coupon_features_raw'])); print('implied versus observed frame fill',np.c_[jj,X[ii,jj]+(inferred[ii]-s[ii])/theta[3:][jj],X[ii,jj]]); print('coeff temporal fixed',B@np.linalg.pinv(np.vstack([K(tau),T]))); print('labrange deployment',np.unique(te['lab_id'],return_counts=True))
