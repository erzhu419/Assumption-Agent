import importlib.util, pickle, numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=d['train']; te=d['test']; coupon,test=m._prepare(d); dt,t=d['past_dt'],d['eval_times']; y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
print('column signal std',np.round(y.std(0),7))
for loss in ('linear','soft_l1','cauchy'):
    p=np.empty_like(y); taus=[]
    for lab in np.unique(tr['lab_id']):
        hold=tr['lab_id']==lab
        tau,w=m._fit(tuple(a[~hold] for a in coupon),dt,t,y[~hold],loss=loss)
        p[hold]=m._design(tuple(a[hold] for a in coupon),dt,t,tau)@w
        taus.append(tau)
    e=abs(p-y)
    for lab in np.unique(tr['lab_id']):
        r=e[tr['lab_id']==lab]; print(loss,'lab',lab,'curve/rate med',np.round([np.median(r[:,:-1]),np.median(r[:,-1])],8),'p90',np.round([np.quantile(r[:,:-1],.9),np.quantile(r[:,-1],.9)],8),'tau',np.round(taus[lab],5))
    print(loss,'by horizon median',np.round(np.median(e,axis=0),8))
out=m.solve_batch(d); print('full material',out['estimated_taus'],out['hereditary_weights']); print('test drift range',np.ptp(test[2]@out['hereditary_weights'][2:]),'predicted rate range',np.ptp(out['initial_rate_pred']))
