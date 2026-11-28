import pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as b
m = runpy.run_path('/holdout_task/downstream_solver.py')
raw = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
d = b._up.calibrate_upstream(raw)
tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']
x=m['_features'](tr,te); n=len(tr['lab_id'])
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; lab=tr['lab_id']
a=m['_inputs'](tr,x[:n])
for split in [None,*np.unique(lab)]:
    fit=np.ones(n,bool) if split is None else lab!=split
    val=fit if split is None else ~fit
    tau,w=m['_fit'](tuple(k[fit] for k in a),dt,t,y[fit],r[fit]); assert .05<tau[0]<tau[1]<25
    s,p=m['_design'](tau,tuple(k[val] for k in a),dt,t)
    e=s@w-y[val]; er=p@w-r[val]
    print('lab', split, 'tau', np.round(tau,5), 'w', np.round(w,6), 'strain/rate absolute q50/q90', np.quantile(abs(e),[.5,.9]),np.quantile(abs(er),[.5,.9]), 'worst time q90',np.quantile(abs(e),.9,axis=0).max())
out=b.solve_batch(raw)
inp=m['_inputs'](te,x[n:]); tau,w=out['estimated_taus'],out['hereditary_weights']
ss,rr=m['_chamber'](tau,w,inp,dt,np.array([0.,1.e-6]))
np.testing.assert_allclose(ss[:,0],0,rtol=0,atol=1e-12)
np.testing.assert_allclose(rr,out['initial_rate_pred'],rtol=0,atol=1e-12)
print('deployment derivative max absolute error', np.max(np.abs(ss[:,1]/1.e-6-rr)))
np.testing.assert_allclose(ss[:,1]/1.e-6,rr,rtol=0,atol=2.e-5)
