import importlib.util, pickle
from itertools import combinations
import numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
cal=d['calibration_load_cells']; x=cal['bridge_raw']; y=cal['true_channel_loads'].sum(1)
a=np.column_stack((np.ones(len(x)),x)); z=(x-x.mean(0))/x.std(0)
pairs=[z[:,i]+sgn*z[:,j] for i,j in combinations(range(4),2) for sgn in (-1,1)]
q=np.column_stack(pairs); folds=np.array_split(np.random.default_rng(52).permutation(len(x)),5)
models={'affine':a,'pairwise-squares':np.column_stack((a,q**2)),'three-way':np.column_stack((a,*[np.prod(z[:,i],axis=1) for i in combinations(range(4),3)]))}
for scale in (.01,.1,1.): models[f'rational-{scale}']=np.column_stack((a,(1/(1+scale*q*q))-1))
for name,m in models.items():
    err=[]
    for f in folds:
        tr=np.setdiff1d(np.arange(len(x)),f)
        err.extend(m[f]@np.linalg.lstsq(m[tr],y[tr],rcond=None)[0]-y[f])
    print(name,'calibration OOF RMSE/max',np.round([np.sqrt(np.mean(np.square(err))),np.max(np.abs(err))],8))
tr,te=d['train'],d['test']; out=s.solve_batch(d); taus=out['estimated_taus']; p,h,f=s._inputs(te,s._features(tr,te)[1],s._bridge(x,cal['true_channel_loads'])); dt,t=d['past_dt'],d['eval_times']; theta=out['hereditary_weights']
for u in (taus,np.array([.05001,24.999])):
    z=np.zeros((len(h),2))
    for j,width in enumerate(dt):
        v=np.exp(-width/u); z=v*z+(1-v)*p[:,j,None]
    c,r=s._design(u,p,h,f,dt,t)
    print('exact recurrence and derivative design max discrepancies',np.max(np.abs(c[:,:,:2]-(h[:,None]-z)[:,None,:]*(1-np.exp(-t[:,None]/u)))),np.max(np.abs(r[:,:2]-(h[:,None]-z)/u)))
    assert np.allclose(c[:,:,:2],(h[:,None]-z)[:,None,:]*(1-np.exp(-t[:,None]/u))) and np.allclose(r[:,:2],(h[:,None]-z)/u)
    eps=1e-6
    cp,_=s._design(u,p,h,f,dt,np.array([eps])); cm,_=s._design(u,p,h,f,dt,np.array([-eps])); assert np.allclose((cp[:,0]@theta-cm[:,0]@theta)/(2*eps),r@theta,rtol=1e-7,atol=1e-8)
# A grouped jackknife is the available nuisance audit for continuous targets.
for l in np.unique(tr['lab_id']):
    mask=tr['lab_id']==l
    sub=dict(d,train={k:v[~mask] for k,v in tr.items()},test={k:v[mask] for k,v in tr.items() if k in te})
    # Retain the unlabeled deployment coupons in the reference and imputation pool.
    sub['test']={k:np.concatenate((te[k],sub['test'][k])) for k in te}
    o=s.solve_batch(sub); n=len(te['lab_id']); ce=o['predicted_strain'][n:]-tr['strain_curves_obs'][mask]; re=o['initial_rate_pred'][n:]-tr['instant_rate_obs'][mask]
    print('held-out lab',l,'signed medians',round(float(np.median(ce)),6),round(float(np.median(re)),6),'trimmed curve RMS/rate RMS',round(float(np.sqrt(np.mean(np.minimum(abs(ce),.04)**2))),6),round(float(np.sqrt(np.mean(re**2))),6),'gross strain/rate counts',int(np.sum(abs(ce)>.08)),int(np.sum(abs(re)>.04)))
