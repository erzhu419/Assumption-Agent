import json,pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
import material_probe as m
item=pickle.load(open('public_batch.pkl','rb'))[0]
raw=item['dataset']; pay=json.load(open('_upstream_param_payload.json'))
d=calibrate_upstream(raw,pay); q=calibrate_upstream(dict(raw,test=item['campaign_diagnostics']),pay)['test']
tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']
x,_=m._features(tr,te); tau,theta=m._material(tr,x,dt,t)
h=q['hold_channel_raw'].sum(-1); p=q['past_channel_raw'].sum(-1)
y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]
b,br=m._basis(tau,t); B=np.vstack((b,br)); state=np.zeros((len(h),2))
for step,ps in zip(dt,p.T): state+=-np.expm1(-step/tau)*(ps[:,None]-state)
mech=((h[:,None]-state)*theta[:2])@B[:,:2].T
miss=np.isnan(q['coupon_features_raw']).any(1)
def score(e):
    return np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],5)
for label,X in [('diagnostic reference panel',m._features(q)[0]),('combined reference panel',m._features(tr,te,q)[-1])]:
    force=X@theta[2:]; base=mech+force[:,None]*B[:,2]; M=B[:,:2]*tau; R=y-base
    G=force[:,None,None]*(M-B[:,2,None])[None,:,:]
    U=force[:,None,None]*np.c_[M,B[:,2]][None,:,:]
    extra=np.concatenate((G, h[:,None,None]*B[None,:,:2], state[:,None,:]*B[None,:,:2]),axis=2)
    print(label,'canonical force vs rate gap',score(base-y),'missing rows',np.sum(miss))
    for kind,A,Z in [('conserved',G,R),('unconstrained',U,y-mech),('extra history/load',extra,R)]:
        cv=np.zeros_like(y); by=[]; coeff=[]
        for lab in np.unique(q['lab_id']):
            test=q['lab_id']==lab; train=~test
            w=np.linalg.lstsq(A[train].reshape(-1,A.shape[2]),Z[train].ravel(),rcond=None)[0]
            cv[test]=A[test]@w-Z[test]; by.append(score(cv[test])); coeff.append(w)
        w=np.linalg.lstsq(A.reshape(-1,A.shape[2]),Z.ravel(),rcond=None)[0]
        print(kind,'full',np.round(w,6),'fold range',np.round([np.min(coeff,0),np.max(coeff,0)],5),'overall',score(cv),'worst labs',np.max(by,0),'missing vs complete',score(cv[miss]),score(cv[~miss]))
    for w in [[.58,.66],[.58,.66+0.02],[.58,.68],[.60,.66],[.60,.68]]:
        print('candidate',w,'rmse',score(G@w-R))
    print('baseline rate residual max',np.max(abs(base[:,-1]-y[:,-1])))
