import sys, pickle, importlib.util
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('previous','/tmp/hegel_continual_archive/alpha_m100_c03_batch_02_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=calibrate_upstream(dict(p['dataset'],chamber_verification_trials=p['campaign_diagnostics']))
tr,te,di=[d[k] for k in ('train','test','chamber_verification_trials')]
dt,t=d['past_dt'],d['eval_times']
load=lambda s: tuple(np.sum(s[k],axis=-1) for k in ('past_channel_raw','hold_channel_raw'))
y=lambda s: s['strain_curves_obs']-s['strain_0'][:,None]
n=len(tr['lab_id']); x=m._features(tr,te)
taus,th=m._fit(x[:n],*load(tr),dt,t,y(tr),tr['instant_rate_obs'])
print('taus',taus,'theta',th)
for name,s,u in [('train',tr,x[:n]),('diag pooled with train',di,m._features(tr,di)[n:]),('diag independently scaled',di,m._features(di,di)[:len(di['lab_id'])])]:
 a,b=m._design(taus,u,*load(s),dt,t)
 r=y(s)-a@th; q=s['instant_rate_obs']-b@th
 print(name,'labs',np.bincount(s['lab_id']),'median absolute curve and rate',np.round(np.median(abs(r),axis=0),6),round(np.median(abs(q)),6),'range',np.round([r.min(),r.max(),q.min(),q.max()],5))
 if s is di:
  print('resid first six',np.round(np.column_stack((q,r[:,[0,3,7,-1]]))[:6],5))
  print('corr',np.round(np.corrcoef(np.column_stack((q,r[:,-1],b,load(s)[0],load(s)[1])).T)[:2,2:],2))
