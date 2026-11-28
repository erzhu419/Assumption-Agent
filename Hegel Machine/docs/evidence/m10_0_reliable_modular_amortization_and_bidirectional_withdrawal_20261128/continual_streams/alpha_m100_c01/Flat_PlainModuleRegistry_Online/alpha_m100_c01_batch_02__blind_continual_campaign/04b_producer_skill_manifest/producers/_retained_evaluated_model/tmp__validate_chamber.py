import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task')
import downstream_solver as m
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl','rb') as f:
    i=pickle.load(f)[0]
d=calibrate_upstream(i['dataset']); s=calibrate_upstream(dict(i['dataset'],test=i['campaign_diagnostics']))['test']
tr=d['train']; dt=d['past_dt']; t=d['eval_times']; xx,_=m._features(tr,d['test']); x,_=m._features(s,s)
Y=s['strain_curves_obs']; R=s['instant_rate_obs']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]
for lab in np.unique(s['lab_id']):
    te=s['lab_id']==lab; tr_fit=tr['lab_id']!=lab
    arrays=m._inputs(tr,xx)
    tau,th=m._fit(*(a[tr_fit] for a in arrays),dt,t,y[tr_fit],tr['instant_rate_obs'][tr_fit]);
    def pred(h):return m._chamber(s,x,dt,t,tau,th,h)
    h=minimize_scalar(lambda h: np.mean((pred(h)[0][~te]-Y[~te])**2), bounds=(.3,3),method='bounded').x
    print('held lab',lab,'tau',np.round(tau,4),'fitted drift time',round(h,5))
    for label, h1 in [('fixed',1.),('fitted',h),('benchtop',1e8)]:
        p,r=pred(h1); print(label, 'curve P90/max',np.round(np.quantile(abs(p[te]-Y[te]),[.9,1]),5),'rate P90/max',np.round(np.quantile(abs(r[te]-R[te]),[.9,1]),5))
