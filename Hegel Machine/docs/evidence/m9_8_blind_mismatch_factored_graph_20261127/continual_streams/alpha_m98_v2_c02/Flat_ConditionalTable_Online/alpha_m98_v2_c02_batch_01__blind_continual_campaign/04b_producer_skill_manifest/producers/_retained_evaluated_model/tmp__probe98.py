import pickle,numpy as np,runpy
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True)
j=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=j['dataset'];a=j['campaign_diagnostics'];m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m98_v2_c02_s00_solver.py');o=m['solve_batch'](d);u=o['estimated_taus'];b=o['hereditary_weights'];t=d['eval_times'];dt=d['past_dt'];st=m['_stress_calibration'](d['calibration_load_cells']);s=st(a['past_channel_raw']);h=st(a['hold_channel_raw']);
stack=lambda C,R:np.c_[C,R*.7]; Y=stack(a['strain_curves_obs']-a['strain_0'][:,None],a['instant_rate_obs']);
def basis(u):return stack(np.c_[-np.expm1(-t[:,None]/u),t].T,np.r_[1/u,1]);
def amplitudes(u):
 H=basis(u); return np.linalg.lstsq(H.T,Y.T,rcond=None)[0].T
opt=least_squares(lambda v:(amplitudes(np.exp(v))@basis(np.exp(v))-Y).ravel(),np.log(u),bounds=(np.log([.05,1]),np.log([2,25])));print('diag free taus',np.exp(opt.x),'RMS',np.sqrt(np.mean(opt.fun**2)),'fixed RMS',np.sqrt(np.mean((amplitudes(u)@basis(u)-Y)**2)));print('svd',np.linalg.svd(Y,compute_uv=False))
xx={k:np.concatenate([d['test'][k],a[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}; x=m['_standardize'](d['train'],xx)[1][-len(h):];f=np.c_[np.ones(len(h)),x]; W=-np.expm1(-dt[:,None]/u)*np.exp(-(np.cumsum(dt[::-1])[::-1]-dt)[:,None]/u); B=amplitudes(u);D=B-np.c_[(h[:,None]-s@W)*b[:2],f@b[2:]];
for name,X in [('past',np.c_[np.ones(len(h)),h,s]),('feat',f),('past+feat',np.c_[f,h,s]),('last',np.c_[np.ones(len(h)),h,s[:,-2:]]),('gap',np.c_[np.ones(len(h)),h[:,None]-s@W]),('past squared',np.c_[np.ones(len(h)),h,s,s[:,-1]**2]),('past + strain0',np.c_[np.ones(len(h)),h,s,a['strain_0']])]:
 P=np.linalg.pinv(X);C=P@D; E=D-X@C;cv=E/(1-np.diag(X@P)[:,None]);print(name,'coeff',C.T,'in rmse',np.sqrt(np.mean(E**2,0)),'loo curve/rate',np.sqrt(np.mean((cv@basis(u))**2,0))[[0,5,11,12]]);
print('drift',np.c_[f@b[2:],B[:,2],D[:,2],D[:,1]]); print('correlation amp residuals with past, hold, x',np.corrcoef(np.c_[D,s,h,x].T)[:3,3:]); print('diag reference/raw first row',a['ref_coupons'][0],a['coupon_features_raw'][0]);
