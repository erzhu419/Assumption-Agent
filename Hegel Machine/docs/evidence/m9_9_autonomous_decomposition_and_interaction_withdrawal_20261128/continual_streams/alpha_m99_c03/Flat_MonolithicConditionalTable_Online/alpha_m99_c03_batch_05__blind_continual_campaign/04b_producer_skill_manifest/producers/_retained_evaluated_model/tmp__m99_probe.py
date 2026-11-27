import pickle, numpy as np, importlib.util
from pathlib import Path
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=x['dataset']; d=x['campaign_diagnostics']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']
def load(path):
 s=importlib.util.spec_from_file_location('candidate',path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
p=Path('/tmp/hegel_continual_archive'); m=load(p/'alpha_m99_c03_batch_04_solver.py')
cal=ds['calibration_load_cells']; b=m._calibration(cal); print('effective calibration',np.round(b,5)); raw=cal['bridge_raw']; loads=cal['true_channel_loads']; j=np.arange(4); e=raw-(b[0]+loads@b[1:5]+b[5]*loads**3+b[6]*loads*loads[:,(j+1)%4]); print('bridge max abs residual quantiles',np.quantile(np.max(abs(e),axis=1),[0,.5,.8,.9,1])); print('raw norm largest',np.sort(np.linalg.norm(raw,axis=1))[-12:]);
y=d['strain_curves_obs']-d['strain_0'][:,None]; r=d['instant_rate_obs']; n=len(tr['lab_id']); xt=m._features(tr,te); tau,th=m._fit(m._coupon_inputs(tr,xt[:n],b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau, theta',tau,th)
xd=m._features(tr,d)[n:]; q=m._coupon_inputs(d,xd,b); past,hold,F=q; drift=F@th[2:]; A,R=m._design(tau,q,dt,t); bas=-np.expm1(-t[:,None]/tau); mem=hold[:,None]-R[:,:2]*tau

def metrics(label,c,v):
 ce=c-y; re=v-r
 def error(a):return (round(np.sqrt(np.mean(a*a)),5),round(np.median(abs(a)),5))
 print(label,error(ce),error(re),'worst lab RMSE',max(error(ce[d['lab_id']==k])[0] for k in np.unique(d['lab_id'])),max(error(re[d['lab_id']==k])[0] for k in np.unique(d['lab_id'])))
metrics('benchtop',A@th,R@th)
import inspect
for f in sorted(p.glob('*solver.py')):
 mm=load(f)
 if not hasattr(mm,'_chamber'):continue
 args=list(inspect.signature(mm._chamber).parameters); vals={'taus':tau,'theta':th,'coupons':q,'hold':hold,'x':xd,'dt':dt,'times':t};
 if set(args)<=vals.keys(): metrics(f.stem,*mm._chamber(*(vals[k] for k in args)))
for basis_name,g in [('none',np.zeros_like(t)),('linear',t),('log',np.log1p(t)),('exp',-np.expm1(-t))]:
 for lam in [0,1]:
  B=(hold[:,None]-lam*mem)*th[:2]; metrics((basis_name,lam),B@bas.T+drift[:,None]*g,B@(1/tau)+(0 if basis_name=='none' else drift))
print('diag scales', 'past',np.quantile(past,[0,.5,1]),'hold',np.quantile(hold,[0,.5,1]),'drift',np.quantile(drift,[0,.5,1])); print('curve median hold-scaled shape',np.round(np.median(y/hold[:,None],axis=0),5))
