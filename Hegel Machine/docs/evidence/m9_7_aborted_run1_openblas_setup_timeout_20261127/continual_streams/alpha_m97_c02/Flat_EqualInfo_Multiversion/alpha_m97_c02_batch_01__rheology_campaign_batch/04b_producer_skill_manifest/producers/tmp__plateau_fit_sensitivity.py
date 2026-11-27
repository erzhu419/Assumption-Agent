import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
p,h,x,y0=s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
train=d['train']; lab=train['lab_id']; n=len(lab)
y=train['strain_curves_obs']-y0[:n,None]; r=train['instant_rate_obs']
complete=np.isfinite(train['coupon_features_raw']).all(axis=1)
for complete_fit in (False,True):
    worst=np.zeros((3,2)); fold_taus=[]
    for j in np.unique(lab):
        tr=np.flatnonzero((lab!=j)&(complete | (not complete_fit)))
        va=np.flatnonzero(lab==j)
        z,w=s._fit(p[tr],h[tr],x[tr],dt,t,a[tr],y[tr],r[tr],True)
        c,b,*_=s._basis(z,p[va],h[va],x[va],dt,t,a[va],True)
        e=c@w-y[va]; er=b@w-r[va]; fold_taus.append(np.exp(z))
        for i,m in enumerate((complete[va],~complete[va],np.ones(len(va),dtype=bool))):
            worst[i]=np.maximum(worst[i], [np.median(abs(e[m]),axis=0).max(),np.median(abs(er[m]))])
        print('complete_fit',complete_fit,'lab',j,'signed residual medians among complete coupons',np.round(np.median(e[complete[va]],axis=0)[[0,-1]],5),round(np.median(er[complete[va]]),5))
    print('complete_fit',complete_fit,'worst medians [complete, missing, all] x [curve, rate]',worst.round(6).tolist(),'tau fold range',np.ptp(fold_taus,axis=0).round(6))
