import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
p,h,x,y0=s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
l=d['train']['lab_id']; n=len(l)
y=d['train']['strain_curves_obs']-y0[:n,None]; r=d['train']['instant_rate_obs']
print('active by lab (active/total):',[(j,int(np.sum((l==j)&(a[:n]>0))),int(np.sum(l==j))) for j in np.unique(l)], 'test',sum(a[n:]>0),len(a)-n)
for full in (False,True):
    for j in [None,*np.unique(l)]:
        tr=np.flatnonzero(l!=j); va=np.flatnonzero(l==j) if j is not None else tr
        z,w=s._fit(p[tr],h[tr],x[tr],dt,t,a[tr],y[tr],r[tr],full_plateau=full)
        cb,rb,*_=s._basis(z,p[va],h[va],x[va],dt,t,a[va],full_plateau=full)
        e=cb@w-y[va]; er=rb@w-r[va]
        print('full',full,'lab',j,'taus',np.exp(z).round(5),'weights',w.round(5), 'median and rmse curve/rate',[(round(np.median(abs(v)),5),round(np.sqrt(np.mean(v**2)),5)) for v in (e,er)], 'medians by damage',[(m,np.round(np.median(abs(e[(a[va]>0)==m]),axis=0)[[0,-1]],5)) for m in (False,True) if np.any((a[va]>0)==m)])
