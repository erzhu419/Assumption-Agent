import pickle,numpy as np,glob,runpy
from scipy.optimize import least_squares
m=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); q=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=q['dataset']; D=q['campaign_diagnostics']; tr,te=d['train'],d['test']; S=m['_stress_calibration'](d['calibration_load_cells']); x,xt=m['_standardize'](tr,te); dt,t=d['past_dt'],d['eval_times']; tau,beta=m['_fit'](m['_design'](tr,x,S,dt,t),tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); h=S(D['hold_channel_raw']); y=np.c_[D['strain_curves_obs']-D['strain_0'][:,None],D['instant_rate_obs']]; xd=m['_standardize'](D,D)[0]; rd=beta[2]+xd@beta[3:]; print('diagnostic labs',np.unique(D['lab_id'],return_counts=True)); print('train/tau/beta',tau,beta)
def modes(v): return np.column_stack((-np.expm1(-t[:,None]/v).T,1/v))
def calc(G,kind,keep):
 def pred(p):
  v=tau if kind!='free_tau' else p[-2:]; B=modes(v)
  if kind=='relative': return (h[:,None]*beta[:2]+G*p[:2])@B
  if kind=='scaled_tau': return (h[:,None]*beta[:2]+G*(v*p[0]))@B
  return (h[:,None]*p[:2]+G*p[2:4])@B
 p={'relative':[.28,2.1],'scaled_tau':[.5],'free_tau':[*beta[:2],.28,2.1,*tau],'free_hold':[*beta[:2],.28,2.1]}[kind]; o=least_squares(lambda p:(pred(p)-y)[keep].ravel(),p); return o.x,pred(o.x)-y
for kind in ('relative','scaled_tau','free_hold','free_tau'):
 G=rd[:,None]; p,e=calc(G,kind,np.ones(len(h),bool)); eo=np.stack([calc(G,kind,np.arange(len(h))!=i)[1][i] for i in range(len(h))]); labs=np.unique(D['lab_id']); el=np.concatenate([calc(G,kind,D['lab_id']!=l)[1][D['lab_id']==l] for l in labs]); print(kind,p,'RMSE strain/rate in, coupon-out, lab-out',[(np.sqrt(np.mean(a[:,:-1]**2)),np.sqrt(np.mean(a[:,-1]**2))) for a in (e,eo,el)],'worst lab',[(int(l),np.sqrt(np.mean(eo[D['lab_id']==l,:-1]**2)),np.max(abs(eo[D['lab_id']==l,-1]))) for l in labs]); print('coupon errors',np.round(np.c_[D['lab_id'],rd,np.sqrt(np.mean(e[:,:-1]**2,axis=1)),e[:,-1]],4))
