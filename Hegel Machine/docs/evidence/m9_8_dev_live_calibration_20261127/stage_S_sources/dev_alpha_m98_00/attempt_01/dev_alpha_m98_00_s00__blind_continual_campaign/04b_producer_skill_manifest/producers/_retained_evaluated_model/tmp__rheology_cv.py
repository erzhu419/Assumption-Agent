import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
a = d['train']; y = a['strain_curves_obs'] - a['strain_0'][:,None]; v = a['instant_rate_obs']
dt, t = d['past_dt'], d['eval_times']; x, _ = s._prepare(d)
for adaptive in (False, True):
    print('adaptive:', adaptive)
    for group in np.unique(a['lab_id']):
        val = a['lab_id'] == group
        taus, theta = s._fit(dt,t,tuple(z[~val] for z in x),y[~val],v[~val],adaptive=adaptive)
        c,r = s._design(taus,dt,t,tuple(z[val] for z in x))
        ce,re = np.abs(c@theta-y[val]),np.abs(r@theta-v[val])
        print(group, 'taus',np.round(taus,5), 'curve med/p90/max-time-p90', np.round([np.median(ce),np.quantile(ce,.9),np.quantile(ce,.9,axis=0).max()],5), 'rate med/p90',np.round(np.quantile(re,[.5,.9]),5))
    print('all-fit:', s._fit(dt,t,x,y,v,adaptive=adaptive))
