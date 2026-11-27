import runpy,pickle,numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/tmp/hegel_continual_archive/dev_alpha_m98_00_s00_solver.py'); p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; q=p['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; o=m['solve_batch'](d); tau=o['estimated_taus']; th=o['hereditary_weights']; cal=m['_stress_calibration'](d['calibration_load_cells']); a=q['past_channel_raw']@cal[:-1]+cal[-1]; b=q['hold_channel_raw']@cal[:-1]+cal[-1]; x=m['_aligned_features'](tr,q)[len(tr['lab_id']):]; A,R=m['_design'](tau,dt,t,(a,b,x)); Y=np.column_stack((q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs'])); E=Y-np.column_stack((A@th,R@th)); n=len(Y); print('estimated intrinsic:',tau,th)
def basis(z, drift=True):
 U=np.vstack((-np.expm1(-t[:,None]/z),1/z))
 return np.column_stack((U,np.r_[t,1])) if drift else U
w=np.r_[np.ones(len(t)),1.]
def project(y,z,drift=True):
 U=basis(np.asarray(z),drift); C=np.linalg.lstsq(w[:,None]*U,(y*w).T,rcond=None)[0].T
 return C,y-C@U.T
for yname,y in [('total',Y),('mismatch',E)]:
 print(yname,'svd',np.linalg.svd(y,compute_uv=False).round(5))
 for q0,fix,drift in [([],tau,True),([.5],[],True),([.2,2.],[],True),([.2,2.],[],False),([.5],tau,True)]:
  z=np.r_[fix,q0];
  if q0:
   opt=least_squares(lambda v:(project(y,np.r_[fix,np.exp(v)],drift)[1]*w).ravel(),np.log(q0),bounds=(np.log(.04),np.log(40.)),max_nfev=500,loss='soft_l1',f_scale=.015); z=np.r_[fix,np.exp(opt.x)]
  C,r=project(y,z,drift); print('modes',z.round(5),'drift',drift,'median/p90',np.quantile(abs(r),[.5,.9]).round(5),'C spreads',np.std(C,0).round(5));
  if (yname=='total' and ((len(z)==2 and not q0) or (len(z)==2 and q0 and drift))) or (yname=='mismatch' and len(z)==1):
   cand=np.column_stack((b,a,x[:,1:],A[:,0,:2]/(-np.expm1(-t[0]/tau)),x@th[2:])); labels=['hold']+['past'+str(i) for i in range(a.shape[1])]+['x'+str(i) for i in range(4)]+['gap0','gap1','base_drift']; print('corr',[(label,np.round(row,2).tolist()) for label,row in zip(labels,np.corrcoef(cand.T,C.T)[:cand.shape[1],cand.shape[1]:])]); print('coef lines',np.round(C[:6],4))
