import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c01_s00_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=p['dataset']; tr=d['train'];te=d['test'];g=p['campaign_diagnostics'];dt=d['past_dt']; t=d['eval_times']; bm=m._axial_load_map(d['calibration_load_cells']);xt,xe=m._features(tr,te); s=m._specimen(tr,xt,bm); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs'];tau,th=m._fit(s,dt,t,y,r)
np.set_printoptions(precision=5,suppress=True,linewidth=180)
def report(e,f):
 print('curve/rate RMSE',np.sqrt((e**2).mean()),np.sqrt((f**2).mean()),'MAD scales',*[1.4826*np.median(abs(a-np.median(a))) for a in (e,f)],'max specimen',np.max(np.sqrt((e**2).mean(1))),np.max(abs(f)))
def predict(s):
 c,b=m._branches(s,dt,t,tau); drift=s[2]@th[2:]; return c@th[:2]+drift[:,None]*t,b@th[:2]+drift
print('tau theta',tau,th,'bridge',bm);cal=d['calibration_load_cells'];print('bridge axial rmse',np.sqrt(np.mean(((cal['bridge_raw']-bm[0])@bm[1]-cal['true_channel_loads'].sum(1))**2)))
a,b=predict(s);print('bench');report(a-y,b-r)
base={k:np.concatenate([tr[k],te[k]]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
_,x=m._features(base,g); Z=m._specimen(g,x,bm);past,hold,F=Z; Y=g['strain_curves_obs']-g['strain_0'][:,None];R=g['instant_rate_obs'];a,b=predict(Z);print('baseline diagnostic');report(a-Y,b-R)
print('bench outliers',np.quantile(abs(a-Y),[.5,.9,.99]),np.quantile(abs(b-R),[.5,.9,.99]))
# Keep observed onset rates as a separate measurement in the temporal basis.
obs=np.c_[Y,R]; w=np.r_[np.ones(len(t)),.35]
def basis(tau): return np.c_[np.r_[-np.expm1(-t/tau[0]),1/tau[0]],np.r_[-np.expm1(-t/tau[1]),1/tau[1]],np.r_[t,1.]]
def prof(logtau):
 B=basis(np.exp(logtau)); A=np.linalg.lstsq(B*w[:,None],(obs*w).T,rcond=None)[0].T;return A,A@B.T-obs
for initial in [tau,tau*[.7,1.4],tau*[1.4,.7]]:
 fit=least_squares(lambda q:(prof(q)[1]*w).ravel(),np.log(initial),bounds=(np.log(.05),np.log(25))); print('free chamber temporal',np.exp(fit.x),fit.cost);e=prof(fit.x)[1];report(e[:,:-1],e[:,-1])
A,e=prof(np.log(tau));print('same temporal basis');report(e[:,:-1],e[:,-1]);c,rr=m._branches(Z,dt,t,tau);expected=np.c_[rr*tau*th[:2],F@th[2:]]; print('amplitude measured vs baseline and deltas');print(np.c_[A,expected,A-expected]);
inputs=np.c_[hold,past,F];print('coefs full additive input basis [hold, past chronological, intercept, features]');print(np.linalg.lstsq(inputs,A,rcond=None)[0].T);print('archive summary',[(j['completed_short_name'],{k:v for k,v in j.items() if k!='facility_telemetry_trace'}) for j in __import__('json').load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))]);
