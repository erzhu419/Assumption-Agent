import pickle,runpy,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/alpha_m98_v2_c03_'
a=runpy.run_path(p+'batch_06_solver.py'); b=runpy.run_path(p+'s00_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; v=i['campaign_diagnostics']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; model=a['_calibration'](d['calibration_load_cells']); print('bridge cross/cubic',model[2:]);
def obs(z): return np.column_stack((z['strain_curves_obs']-z['strain_0'][:,None],z['instant_rate_obs']))
def score(e): return np.round([np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),np.quantile(abs(e[:,:-1]),.9),np.quantile(abs(e[:,-1]),.9)],6)
for m in (a,b):
 c,_=m['_prepare'](d); u,w=m['_fit'](c,dt,t,obs(tr)); r=m['_design'](c,dt,t,u)@w-obs(tr); print('train',m['__file__'],u,w,score(r))
u,w=a['_fit'](a['_prepare'](d)[0],dt,t,obs(tr)); y=obs(v)
for ref in ('train','test','self'):
 if ref=='self': z=a['_features'](v,v)[:len(y)]
 else: z=a['_features'](d[ref],v)[-len(y):]
 c=a['_coupon'](v,z,model); h=c[1]; f=c[2]@w[2:]; elastic=a['_design']((c[0],h,np.zeros_like(c[2])),dt,t,u)@w; r=np.vstack((-np.expm1(-t[:,None]/u),1/u)); reset=(h[:,None]*w[:2])@r.T; lin=np.r_[t,1.]; exp1=np.r_[-np.expm1(-t),1.];
 cand={'baseline':elastic+f[:,None]*lin,'drift_exp1':elastic+f[:,None]*exp1,'reset_linear':reset+f[:,None]*lin,'reset_01':reset+f[:,None]*(r@np.array([.19694,2.09244])),'elastic_only':elastic,'reset_only':reset};
 c06,rr=a['_chamber_predict'](c,t,u,w); cand['batch06']=np.column_stack((c06,rr))
 print(ref,[(k,score(x-y)) for k,x in cand.items()]);
 basis=np.stack([elastic,reset,f[:,None]*lin,f[:,None]*exp1,f[:,None]*r[:,0],f[:,None]*r[:,1]],axis=-1)
 for cols in ([0,1,2,3],[0,1,4,5],[0,1,2,4,5]):
  A=basis[:,:,cols]; fit=least_squares(lambda q:(A@q-y).ravel(),np.zeros(len(cols)),loss='soft_l1',f_scale=.01); print('flex',cols,fit.x.round(5),score(A@fit.x-y))
