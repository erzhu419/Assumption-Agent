import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_m97_c02_batch_01_solver as s
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
base = s._calibrate(c)
for j in range(4):
    rows = np.flatnonzero(c['lab_id'] == j)
    for mode in ('one_point', 'one_spike'):
        altered = {k: np.array(v, copy=True) for k,v in c.items()}
        if mode == 'one_point':
            take = np.ones(len(c['lab_id']), dtype=bool)
            take[rows[1:]] = False
            altered = {k:v[take] for k,v in altered.items()}
        else:
            altered['true_stress'][rows[0]] += 4.
        try:
            candidate = s._calibrate(altered)
            print('calibration', j, mode, 'delta gain/bias', (candidate[j]-base[j]).round(7), 'other channels unchanged', np.array_equal(np.delete(candidate,j,axis=0),np.delete(base,j,axis=0)))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            print('calibration', j, mode, type(exc).__name__, str(exc))
p,h,x,y0 = s._prepare(d)
dt,t=d['past_dt'],d['eval_times']
a=s._activity(d,p,dt)
train=d['train']; l=train['lab_id']; n=len(l)
y=train['strain_curves_obs']-y0[:n,None]; r=train['instant_rate_obs']
missing=np.isnan(train['coupon_features_raw']).any(axis=1)
for scales in ((.025,.025),(.018,.018),(.018,.030)):
    maxima=np.zeros(6); pars=[]
    for j in np.unique(l):
        tr=np.flatnonzero(l!=j); va=np.flatnonzero(l==j)
        z,w=s._fit(p[tr],h[tr],x[tr],dt,t,a[tr],y[tr],r[tr],True,scales=scales)
        pars.append(np.r_[np.exp(z),w])
        cb,rb,*_=s._basis(z,p[va],h[va],x[va],dt,t,a[va],True)
        e=abs(cb@w-y[va]); er=abs(rb@w-r[va]); active=a[va]>0
        for masks, start in (([active,~active],0),([missing[va],~missing[va]],2)):
            for m in masks:
                if m.any():
                    maxima[start:start+2]=np.maximum(maxima[start:start+2],[np.median(e[m],axis=0).max(),np.median(er[m])])
        maxima[4:6]=np.maximum(maxima[4:6],[np.median(e[:,-1]),np.median(er)])
    print('scales',scales,'worst errors [damage curve/rate, missingness curve/rate, late lab curve/lab rate]',maxima.round(6),'tau fold ranges',np.ptp(pars,axis=0)[:2].round(6))
