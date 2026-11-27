import pickle, runpy
import numpy as np
from scipy.optimize import least_squares
s = runpy.run_path('/holdout_task/batch_solver.py')
i = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d, v = i['dataset'], i['campaign_diagnostics']
cal = d['calibration_load_cells']; x, y = cal['true_channel_loads'], cal['bridge_raw']
keep = np.max(abs(y-np.median(y,axis=0)),axis=1)<8; x,y=x[keep],y[keep]
for name in ('adjacent+cubic', 'rational interaction', 'rational cubic'):
    errors=[]
    for fold in range(4):
        test=np.arange(len(x))%4==fold
        for j in range(4):
            q,r=x[:,j],x[:,(j+1)%4]
            base=np.column_stack((np.ones(len(x)),x,q*r,q**3))
            p0=np.linalg.lstsq(base[~test],y[~test,j],rcond=None)[0]
            pred=base@p0
            if name!='adjacent+cubic':
                col, z = (5,(q-r)**2) if name=='rational interaction' else (6,q*q)
                def predict(p):
                    return base@p[:-1]+p[col]*base[:,col]*(1/(1+p[-1]*z)-1)
                p=least_squares(lambda p:(predict(p)-y[:,j])[~test],np.r_[p0,0.],bounds=(np.r_[np.full(7,-np.inf),0.],np.full(8,np.inf)),gtol=1e-12).x
                pred=predict(p)
            errors.extend(abs((pred-y[:,j])[test]))
    print(name,'held-out raw p90/max',np.quantile(errors,.9),max(errors))
train=d['train']; dt,t=d['past_dt'],d['eval_times']; train_coupon,_=s['_prepare'](d)
def obs(data): return np.column_stack((data['strain_curves_obs']-data['strain_0'][:,None],data['instant_rate_obs']))
y,z=obs(train),obs(v)
vc=s['_coupon'](v,s['_features'](v,v)[:len(z)],s['_calibration'](cal))
for lab in np.unique(v['lab_id']):
    keep=train['lab_id']!=lab; ix=v['lab_id']==lab
    taus,theta=s['_fit'](tuple(q[keep] for q in train_coupon),dt,t,y[keep])
    strain,rate=s['_chamber_predict'](vc,dt,t,taus,theta)
    err=(np.column_stack((strain,rate))-z)[ix]
    print('lab',lab,'n',sum(ix),'median signed strain/rate',np.median(err[:,:-1]),np.median(err[:,-1]),'p90 absolute strain/rate',np.quantile(abs(err[:,:-1]),.9),np.quantile(abs(err[:,-1]),.9))
