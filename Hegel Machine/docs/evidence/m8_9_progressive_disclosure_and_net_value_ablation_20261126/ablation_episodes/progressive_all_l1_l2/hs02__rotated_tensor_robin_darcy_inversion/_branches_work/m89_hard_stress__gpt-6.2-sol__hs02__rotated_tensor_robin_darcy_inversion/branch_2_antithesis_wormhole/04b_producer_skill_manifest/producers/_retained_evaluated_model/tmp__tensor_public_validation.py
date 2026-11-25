import pickle,sys
import numpy as np
sys.path.insert(0,'/holdout_task')
from tensor_darcy_robin import infer_tensor_darcy_sources, _operator, _responses, _robust_sources
with open('/holdout_task/public_tensor_cases.pkl','rb') as f:
    cases=pickle.load(f)['cases']

def errors(w, offset, z, g):
    field=z[...,0]+z[...,1:]@w
    return (np.linalg.norm(w-g['source_weights'])/np.linalg.norm(g['source_weights']),
            abs(offset-g['sensor_offset']), np.max(np.abs(field-g['u_true'])))

for i,case in enumerate(cases):
    p,g=case['problem'],case['gt']
    z=_responses(p); ij=tuple(p['sensor_ij'].T); X=z[ij][:,1:]; y=p['y_obs']-z[ij][:,0]
    a=p['tikhonov_alpha']; n,m=X.shape
    result=infer_tensor_darcy_sources(p)
    assert set(result)=={'source_weights','sensor_offset','predicted_field'}
    w,o,u=[result[k] for k in ('source_weights','sensor_offset','predicted_field')]
    assert w.shape==(m,) and type(o) is float and u.shape==p['k_xx'].shape and np.all(np.isfinite(u))
    A,b,area=_operator(p); rhs=b+area*(w@p['basis'].reshape(m,-1))
    residual=np.linalg.norm(A@u.ravel()-rhs)/np.linalg.norm(rhs)
    assert residual<1e-10
    print('case',i,'public errors',errors(w,o,z,g),'relative FV residual',residual)
    clean=np.abs(y-X@g['source_weights']-g['sensor_offset'])<0.1
    loo=[]
    for k in range(n):
        keep=np.arange(n)!=k
        ww,oo=_robust_sources(X[keep],y[keep],a)
        loo.append((*errors(ww,oo,z,g),abs(y[k]-X[k]@ww-oo)))
    loo=np.array(loo)
    print('LOSO max errors/error on clean held-out sensor',loo[:,:3].max(axis=0),loo[clean,3].max())
    centered=X-X.mean(axis=0)
    U=np.linalg.svd(centered,full_matrices=False)[0]
    rng=np.random.default_rng(77+i)
    stip=[]
    for trial in range(48):
        k=round(n*.1)
        side=U[:,trial % m] * (-1 if trial % 2 else 1)
        contaminated=np.argsort(side)[-k:] if trial%3==0 else rng.choice(n,k,replace=False)
        noise=rng.normal(0,.0015,n)
        yy=X@g['source_weights']+g['sensor_offset']+noise
        yy[contaminated]+=rng.uniform(1.5,4,k)*(1 if trial%3==0 else rng.choice([-1,1],k))
        ww,oo=_robust_sources(X,yy,a)
        stip.append(errors(ww,oo,z,g))
    print('random and high-leverage 10% stress max errors',np.max(stip,axis=0))
