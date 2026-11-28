import pickle,numpy as np,importlib.util
from pathlib import Path
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('old',next(Path('/tmp/hegel_continual_archive').glob('*solver.py'))); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=p['dataset'];v=p['campaign_diagnostics']; o=m.solve_batch(d); tau=o['estimated_taus']; w=o['hereditary_weights']; dt=d['past_dt'];t=d['eval_times'];x=m._features(v,d['test'])[:len(v['lab_id'])]; inp=m._inputs(v,x,m._calibration(d['calibration_load_cells'])); past,hold,cov=inp
Y=np.column_stack((v['strain_curves_obs']-v['strain_0'][:,None],v['instant_rate_obs'])); D,R=m._design(tau,inp,dt,t); res=Y-np.column_stack((D@w,R@w)); np.set_printoptions(precision=4,suppress=True,linewidth=140)
def basis(ts):
 return np.column_stack((np.vstack((-np.expm1(-t[:,None]/ts),1/ts)),np.r_[t,1.]))
def fit(ts,y):
 A=basis(np.asarray(ts)); c=np.linalg.lstsq(A,y.T,rcond=None)[0].T; return y-c@A.T,c
for name,y in [('full',Y),('increment',res)]:
 print(name,'singular',np.linalg.svd(y,compute_uv=False)[:9]); print('fixed taus',np.quantile(abs(fit(tau,y)[0]),[.5,.9,.98,1]));
 for st in [[.5],[.3,2],[.7,4],[.3,1,5]]:
  q=least_squares(lambda u:fit(np.exp(u),y)[0].ravel(),np.log(st),bounds=(np.log(.05),np.log(25.)),loss='soft_l1',f_scale=.015,max_nfev=120); ts=np.exp(q.x); print('free',ts,np.quantile(abs(fit(ts,y)[0]),[.5,.9,.98,1]))
 _,c=fit(tau,y); print('fixed coefficients mean/std',c.mean(0),c.std(0)); print('coeff rows',c[:6]);
F=np.c_[hold,past,x,cov@w[2:],abs(hold),hold**2,hold[:,None]*x]; print('correlations rows hold,past,raw standardized features,drift,abs hold,sq hold,hold*features',np.corrcoef(F.T,fit(tau,res)[1].T)[:-3,-3:]);print('residual sample',res[:5]); print('hold range',hold.min(),hold.max()); print('diagnostic ref fit vs pooled train',abs(m._features(d['train'],v)[-len(x):]-x).mean(0))
