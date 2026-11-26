import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w03_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w03_s00.pkl','rb'))[0]['dataset']; a=d['train']; x,h,p=m._prep(d); dt,t=d['past_dt'],d['eval_times']; y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; n=len(y); x,h,p=x[:n],h[:n],p[:n]
from itertools import product
complete=np.isfinite(a['coupon_features_raw']).all(axis=1)
for col in ['lab_id','stratum']:
 for i in np.unique(a[col]):
  v=a[col]==i
  for mode in ('all','complete'):
   f=~v if mode=='all' else ~v & complete
   tau,w=m._fit(x[f],h[f],p[f],dt,t,y[f],r[f]); c,b,*_=m._design(tau,x[v],h[v],p[v],dt,t)
   ce=np.abs(c@w-y[v]); re=np.abs(b@w-r[v]); vc=complete[v]
   print(col,i,mode,'tau',tau.round(5),'all curve/rate med',np.round([np.median(ce),np.median(re)],6),'complete worst horizon med/rate p90',np.round([np.max(np.median(ce[vc],axis=0)),np.quantile(re[vc],.9)],6))
