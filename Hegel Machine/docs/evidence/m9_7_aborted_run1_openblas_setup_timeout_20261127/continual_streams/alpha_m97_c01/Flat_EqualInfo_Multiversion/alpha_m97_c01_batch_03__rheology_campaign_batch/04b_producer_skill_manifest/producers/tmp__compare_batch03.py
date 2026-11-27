import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_m97_c01_batch_03_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i,=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb')); d=i['dataset']; g=i['gt']; a=d['train']; b=d['test']; n=len(a['lab_id']); dt=d['past_dt']; t=d['eval_times']; state=m._prepare(d)
y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; complete=np.isfinite(a['coupon_features_raw']).all(1)
train=lambda idx: tuple(v[:n][idx] for v in state)
rmse=lambda x: np.sqrt(np.mean(x*x))
for c in (*np.unique(a['lab_id']),None):
    keep=np.ones(n,bool) if c is None else a['lab_id']!=c
    for name,sel in [('all',keep),('complete',keep&complete)]:
        tau,w=m._fit(train(sel),dt,t,y[sel],r[sel]);
        if c is None:
            A,B=m._design(tau,tuple(v[n:] for v in state),dt,t); e=A@w+b['strain_0'][:,None]-g['te_clean']; f=B@w-g['te_true_rate0'];
            mx=lambda z: max(rmse(z[k]) for lab in np.unique(b['lab_id']) for st in np.unique(b['stratum']) if (k:=(b['lab_id']==lab)&(b['stratum']==st)).any()); print('public',name,'tau',tau,'weight deviations',max(abs(w[:7]-g['theta'])),max(abs(w[7:]-g['vp_theta'])),'rmse',rmse(e),rmse(f),'worst lab/stratum',mx(e),mx(f))
        else:
            A,B=m._design(tau,train(~keep),dt,t); e=A@w-y[~keep]; f=B@w-r[~keep]; print('LOLO',c,name,'curve median absolute',np.median(abs(e)),'rate median absolute',np.median(abs(f)))
