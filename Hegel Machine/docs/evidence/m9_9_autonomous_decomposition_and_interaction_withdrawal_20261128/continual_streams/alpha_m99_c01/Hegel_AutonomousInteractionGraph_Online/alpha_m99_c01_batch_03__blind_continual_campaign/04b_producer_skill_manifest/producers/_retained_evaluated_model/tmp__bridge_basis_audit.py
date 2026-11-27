import pickle, sys
from itertools import combinations, product
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_bridge, _forward, _inverse

for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    d=item['dataset']; c=d['calibration_load_cells']
    x=c['true_channel_loads']; y=c['bridge_raw']
    print('calibration fields:', sorted(c))
    axes=np.count_nonzero(x,axis=1)==1
    tail=np.max(np.abs(x),axis=1)>=1.4
    folds=np.random.default_rng(30).permutation(len(x))%5
    splits=[(f'fold {j}',folds==j) for j in range(5)] + [(f'axis {j}',axes & (x[:,j]!=0)) for j in range(4)] + [('high-load tail',tail)]
    # Explicit pairwise sums/differences: powers up to cubic are already
    # included in the full polynomial comparison. Here test rational curvature.
    pairs=[x[:,i]+s*x[:,j] for (i,j),s in product(combinations(range(4),2),(-1,1))]
    q=np.column_stack([x,*pairs]); rational=q/(1+0.5*q*q)
    nxt=(np.arange(4)+1)%4
    stats=[]
    for name,held in splits:
        train=~held; b=_fit_bridge(x[train],y[train]); yp=_forward(x,b)
        residual=np.linalg.norm(yp-y,axis=1)
        keep=train & (residual<=max(1e-6,8*np.median(residual[train])))
        # The independent observation-scale gap labels held-out spike rows.
        clean=held & (np.linalg.norm(y,axis=1)<10)
        err0=[]; err1=[]; scale=[]
        for j in range(4):
            a=np.column_stack((np.ones(len(x)),x,x[:,j]**3,x[:,j]*x[:,nxt[j]]))
            # Test small identifiable alternatives one at a time, choosing a
            # challenger inside each training fold by leave-one-out residual.
            base=np.linalg.lstsq(a[keep],y[keep,j],rcond=None)[0]
            options=[(np.inf,0,a,base)]
            for k,term in enumerate(rational.T):
                z=np.column_stack((a,term)); u=z[keep]
                beta=np.linalg.lstsq(u,y[keep,j],rcond=None)[0]
                h=np.einsum('ij,ji->i',u,np.linalg.pinv(u))
                cv=np.mean(((y[keep,j]-u@beta)/np.maximum(1-h,1e-9))**2)
                options.append((cv,k+1,z,beta))
            _,_,z,beta=min(options,key=lambda v:v[0]); err0.append(a[clean]@base-y[clean,j]); err1.append(z[clean]@beta-y[clean,j]); scale.append(beta[-1])
        r0=np.linalg.norm(np.array(err0),axis=0); r1=np.linalg.norm(np.array(err1),axis=0)
        le=np.max(np.abs(_inverse(y[clean],b)-x[clean])); stats.append((r0.max(),r1.max(),le))
        print(name,'bridge max baseline / rational:',*(f'{v:.6g}' for v in (r0.max(),r1.max())),'load max:',f'{le:.6g}')
    print('worst:', np.max(stats,axis=0))
    b=_fit_bridge(x,y); r=np.linalg.norm(_forward(x,b)-y,axis=1)
    print('spikes:',np.flatnonzero(r>1).tolist(),'largest ordinary residual:',r[r<=1].max())
    print('calibration nuisance fields:', [k for k in c if any(s in k.lower() for s in ('batch','cohort','lab','label'))])
