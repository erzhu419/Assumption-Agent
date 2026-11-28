import numpy as np, pickle, runpy
from scipy.optimize import minimize_scalar, least_squares
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr,te=d['train'],d['test']; dg=p['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; m=runpy.run_path('/tmp/hegel_continual_archive/alpha_m100_c03_s00_solver.py')
b=m['_bridge'](d['calibration_load_cells']); x=m['_features'](tr,te); X=m['_features'](tr,dg)[len(tr['lab_id']):]; past,hold=m['_loads'](tr,b); tau,w=m['_fit'](x[:len(hold)],past,hold,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); past,hold=m['_loads'](dg,b); C,R=m['_design'](tau,X,past,hold,dt,t); Y=np.c_[dg['strain_curves_obs']-dg['strain_0'][:,None],dg['instant_rate_obs']]; E=Y-np.c_[C@w,R@w]; L=np.r_[t,1]; Q=np.r_[t*t,0]
def f(s): return np.r_[-np.expm1(-t/s),1/s]
def fit(A,y=E):
 A=np.column_stack(A); co=np.linalg.lstsq(A,y.T,rcond=None)[0].T; return (co@A.T-y),co
for label,y in [('response',Y),('baseline residual',E)]:
 print(label,'sv',np.linalg.svd(y,compute_uv=False)[:9])
 for name,A in [('nominal', [f(tau[0]),f(tau[1]),L]),('nominal+quadratic',[f(tau[0]),f(tau[1]),L,Q]),('fast+linear',[f(tau[0]),L]),('fast+quad',[f(tau[0]),Q])]:
  e,c=fit(A,y); print(name,np.sqrt(np.mean(e**2)),np.max(abs(e)),np.linalg.svd(c,compute_uv=False))
 for name,bas in [('F+L',lambda s:[f(s[0]),L]),('F+Q',lambda s:[f(s[0]),Q]),('FF',lambda s:[f(s[0]),f(s[1])]),('FF+L',lambda s:[f(s[0]),f(s[1]),L]),('FF+LQ',lambda s:[f(s[0]),f(s[1]),L,Q])]:
  sol=min((least_squares(lambda z:fit(bas(np.exp(z)),y)[0].ravel(),np.log(v),bounds=(-4,5)) for v in [[a,b] if 'FF' in name else [a] for a in (.2,.6,1.5) for b in (1,4)]),key=lambda z:z.cost); s=np.exp(sol.x); e,c=fit(bas(s),y); A=np.column_stack(bas(s)); print(name,s,'rmse/max',np.sqrt(np.mean(e**2)),np.max(abs(e)),'coef sv',np.linalg.svd(c,compute_uv=False),'rate/late rmse',np.sqrt(np.mean(e[:,[-1,-2]]**2,axis=0)))
np.set_printoptions(precision=4,suppress=True,linewidth=130)
P=np.c_[hold,past,X,R[:,2:]@w[2:]]; print('pred columns labels hold,p0..5,x0..3,drift'); print('OLS coeff along response (t first, middle, last, rate):',np.linalg.lstsq(np.c_[np.ones(len(P)),P[:,:-1]],E[:,[0,5,-2,-1]],rcond=None)[0]); print('residue after linear predictors',np.linalg.svd(fit(np.c_[np.ones(len(P)),P[:,:-1]].T, E.T)[0],compute_uv=False)[:5]); print('last stress/rate/last residual:',np.c_[past[:,-1],E[:,-1],E[:,-2]])
for lab in np.unique(dg['lab_id']):
 print('lab',lab,'panel (center, scale)',[(round(np.median(a.mean(0)),5),round(np.median(a.std(0)),5)) for a in [q['ref_coupons'][q['lab_id']==lab] for q in (tr,te,dg)]]); print('diagnostic feature missing',np.sum(np.isnan(dg['coupon_features_raw'][dg['lab_id']==lab]),axis=0))
