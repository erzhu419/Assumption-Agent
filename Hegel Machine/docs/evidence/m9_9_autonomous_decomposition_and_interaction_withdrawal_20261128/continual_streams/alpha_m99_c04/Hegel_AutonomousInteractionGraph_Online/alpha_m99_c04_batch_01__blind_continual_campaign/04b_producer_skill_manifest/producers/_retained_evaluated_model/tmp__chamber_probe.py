import pickle,runpy,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=140)
a=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c04_s00_solver.py'); i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; xt,xe=a['_features'](tr,te); _,x=a['_features'](tr,d); s=a['_stress_map'](ds['calibration_load_cells']); tau,b=a['_fit'](tr,xt,s,dt,t)
print('intrinsic',tau,b,'cal map',s)
for name,g,z in [('train',tr,xt),('chamber',d,x)]:
 c,r=a['_design'](g,z,s,dt,t,tau); yc=g['strain_curves_obs']-g['strain_0'][:,None]; yr=g['instant_rate_obs']; print(name,'n=',len(z),'curve/rate residual percentiles',*[np.percentile(abs(y),[25,50,75,90,99]) for y in (c@b-yc,r@b-yr)])
 if name=='train': sc=np.array([a['_mad']((c@b-yc).ravel()),a['_mad'](r@b-yr)])
for g in (tr,te):
 v=g['aux_chamber_telemetry']; print('aux',v.shape,np.quantile(v,[0,.25,.5,.75,1],axis=0),'corr',np.corrcoef(v.T)); print(v[:3])
past=s[0]+d['past_channel_raw']@s[1:]; hold=s[0]+d['hold_channel_raw']@s[1:]; dr=b[2]+x@b[3:]; y=np.column_stack([yc,yr]); print('noise',sc); print('ranges',np.quantile(np.c_[past,hold,dr],[0,.5,1],axis=0))
scale=np.r_[np.full(len(t),max(sc[0],1e-6)),max(sc[1],1e-6)]; n=len(x)
def basis(tau): return np.vstack([np.column_stack([-np.expm1(-t[:,None]/tau),t]),np.r_[1/tau,1]])
def pred(p): return p[2:].reshape(n,3)@basis(np.exp(p[:2])).T
def res(p): return ((pred(p)-y)/scale).ravel()
for poles in [tau,tau*[.6,1.4],tau*[1.25,.9]]:
 co=np.linalg.lstsq(basis(poles)/scale[:,None],(y/scale).T,rcond=None)[0].T
 f=least_squares(res,np.r_[np.log(poles),co.ravel()],loss='soft_l1',f_scale=2,x_scale='jac',max_nfev=150)
 print('cost/poles/error',f.cost,np.exp(f.x[:2]),np.quantile(abs(pred(f.x)-y),[.5,.9],axis=0))
 if 'best' not in globals() or f.cost<best.cost: best=f
co=best.x[2:].reshape(n,3); print('coefficients vs intrinsic'); print(np.c_[co,dr,hold,past.mean(axis=1),d['lab_id'],d['stratum']]); print('drift regression',np.linalg.lstsq(np.c_[np.ones(n),dr,hold,past.mean(axis=1)],co[:,2],rcond=None)[0]); np.savez('/tmp/chamber_probe.npz',tau=tau,b=b,sc=sc,t=t,dt=dt,x=x,y=y,hold=hold,past=past,dr=dr,co=co,poles=np.exp(best.x[:2]))
