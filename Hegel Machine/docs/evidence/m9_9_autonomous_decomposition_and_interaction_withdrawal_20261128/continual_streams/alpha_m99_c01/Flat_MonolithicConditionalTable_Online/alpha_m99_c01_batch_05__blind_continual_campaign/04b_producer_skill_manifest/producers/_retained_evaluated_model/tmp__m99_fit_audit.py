exec(open('/tmp/m99_check_candidate.py').read().split('out =')[0])
from scipy.optimize import least_squares
tr, te = ds['train'], ds['test']; t, dt = ds['eval_times'], ds['past_dt']; x, _ = m._features(tr, te); br = m._bridge(ds['calibration_load_cells']); p,h = m._loads(tr,br)
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]; r = tr['instant_rate_obs']; tau,beta=m._fit(x,p,h,dt,t,y,r); A,B=m._design(tau,x,p,h,dt,t)
for lab in np.unique(tr['lab_id']):
 s=tr['lab_id']==lab
 for name,e in [('curve',(A@beta-y)[s]),('rate',(B@beta-r)[s])]: print(lab,name,'median/abs quantiles',np.median(e),np.quantile(abs(e),[.5,.8,.95,1]))
print('curve residual median by time',np.median(A@beta-y,axis=0)); print('curve abs median by time',np.median(abs(A@beta-y),axis=0)); print('rate minus curve leading residual correlation by time',(B@beta-r)@np.clip(A@beta-y,-.05,.05))
