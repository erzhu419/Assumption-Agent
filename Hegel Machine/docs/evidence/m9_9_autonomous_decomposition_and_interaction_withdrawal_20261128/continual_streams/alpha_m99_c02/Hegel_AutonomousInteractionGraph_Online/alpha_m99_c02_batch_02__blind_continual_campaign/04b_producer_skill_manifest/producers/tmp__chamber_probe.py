import pickle,runpy,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
s=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_s00_solver.py'); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; g=item['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; x,xt=s['_features'](tr,te); b=s['_bridge'](d['calibration_load_cells']); a,h=s['_material'](tr,b,x,dt,t)
if g is None: raise SystemExit('blind')
_,X=s['_features'](tr,g); p,l=s['_loads'](g,b); Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; F,R=s['_basis'](a,p,l,X,dt,t); base=np.c_[F@h,R@h];
def temporal(tau):
 return np.r_[np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1][None,:]]
def amplitudes(tau):
 T=temporal(tau); return np.linalg.lstsq(T,Y.T,rcond=None)[0].T
for seed in ([.65,3],[.5,6],[.8,12]):
 fit=least_squares(lambda u:(amplitudes(np.exp(u))@temporal(np.exp(u)).T-Y).ravel(),np.log(seed),bounds=(np.log([.05,.05]),np.log([25,40]))); q=np.exp(fit.x);print('chamber apparent taus',q,'column rms',np.sqrt(np.mean(fit.fun.reshape(Y.shape)**2,axis=0)))
print('base apparent taus',a,'free amps column rms',np.sqrt(np.mean((amplitudes(a)@temporal(a).T-Y)**2,axis=0)))
gap=R[:,:2]*a; drift=R[:,2:]@h[2:]; A=amplitudes(q); print('rows: hold,past end, feature drift, gap1, gap2, A1,A2,drift'); print(np.c_[l,p[:,-1],drift,gap,A]); print('corr base/correction'); print(np.corrcoef(np.c_[p,l,X,gap,drift,A,np.subtract(A,np.c_[gap*h[:2],drift])].T)[-6:,:14]);
for k,v in d['calibration_load_cells']['single_channel_sweeps'].items():print(k,v[:12])
for label in np.unique(tr['lab_id']):
 for name,z in [('tr',tr),('te',te),('diag',g)]:
  r=z['ref_coupons'][z['lab_id']==label];print(label,name,np.median(r.mean(0)),np.median(r.std(0)))
