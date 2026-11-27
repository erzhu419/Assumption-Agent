import importlib.util,pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
tr=d['train']; ch=d['chamber_verification_trials']; te=d['test']; t=d['eval_times']

def subset(block, mask):
    return {k:v[mask] for k,v in block.items()}

def report(label,b,e):
    groups={'all':np.ones(len(e),bool),'missing':np.isnan(b['coupon_features_raw']).any(axis=1),'complete':~np.isnan(b['coupon_features_raw']).any(axis=1)}
    groups.update({f'lab_{v}':b['lab_id']==v for v in np.unique(b['lab_id'])})
    print(label,{k:np.round([np.mean(abs(e[m,:-1])), np.mean(abs(e[m,-1])),np.mean(abs(e[m,-2]))],5).tolist() for k,m in groups.items() if m.any()})

def eval_folds(b,labels,pilot):
    result=np.zeros_like(s._target(b)); base=result.copy()
    for label in np.unique(labels):
        m=labels==label; target=subset(b,m)
        test={k:v for k,v in target.items() if k not in ('strain_curves_obs','instant_rate_obs')}
        q=dict(d,train=tr if pilot else subset(b,~m),test=test,chamber_verification_trials=subset(b,~m) if pilot else None)
        for buf,challenge in ((result,q),(base,dict(q,chamber_verification_trials=None))) if pilot else ((result,q),):
            out=s.solve_batch(challenge)
            tau=out['estimated_taus']; assert .05<tau[0]<tau[1]<25
            buf[m]=np.c_[out['predicted_strain']-target['strain_curves_obs'],out['initial_rate_pred']-target['instant_rate_obs']]
    return result,base

print('MAE order: curve, initial rate, endpoint')
report('benchtop leave-lab-out (no pilot)',tr,eval_folds(tr,tr['lab_id'],False)[0])
for name,labels in [('coupon',np.arange(len(ch['lab_id']))),('stratum',ch['stratum']),('lab',ch['lab_id'])]:
    a,b=eval_folds(ch,labels,True);report('chamber leave-'+name+'-out',ch,a);report('matched no-pilot',ch,b)

# A different affine bridge presentation must not alter any physical prediction.
out=s.solve_batch(d)
M=np.array([[1.25,.14,-.05,.08],[.05,.71,.12,-.05],[.06,-.12,1.57,.06],[-.1,.09,.17,.9]])
u=np.array([.3,-.5,.1,.45]); cal=dict(d['calibration_load_cells']); cal['bridge_raw']=cal['bridge_raw']@M+u
alt=dict(d,calibration_load_cells=cal)
for k in ('train','test','chamber_verification_trials'):
    alt[k]=dict(d[k]);
    for key in ('past_channel_raw','hold_channel_raw'):
        alt[k][key]=d[k][key]@M+u
new=s.solve_batch(alt); diff={k:float(np.max(abs(out[k]-new[k]))) for k in out}; print('bridge re-encoding max absolute changes',diff)
assert max(diff.values()) < 1e-5
age=np.r_[np.cumsum(d['past_dt'][::-1])[::-1],0.]; tau=out['estimated_taus']; theta=out['hereditary_weights']; x=np.ones((3,5)); load=(np.arange(18).reshape(3,6)*.3,np.array([1.,3.,-2.]))
y=s._design(load,x,tau,age,np.array([0.,1e-6]))@theta
assert np.allclose(y[:,0],0) and np.allclose(y[:,1]/1e-6,y[:,-1],atol=1e-5)
print('zero-time initial condition and analytical rate verified')
