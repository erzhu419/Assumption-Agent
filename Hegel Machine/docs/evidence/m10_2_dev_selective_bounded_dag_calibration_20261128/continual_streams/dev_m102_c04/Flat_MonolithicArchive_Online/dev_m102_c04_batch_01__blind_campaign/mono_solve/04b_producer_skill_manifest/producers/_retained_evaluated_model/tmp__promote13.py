import baseline13 as base, expanded13 as new
import numpy as np,pickle,pathlib
from threadpoolctl import threadpool_limits
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; perm=np.random.default_rng(21).permutation(72)
splits=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(perm[:52],perm[52:])]
err=[]
with threadpool_limits(limits=1,user_api='blas'):
 for tr,te in splits:
    q={**s,'train':{**d,**{k:d[k][tr] for k in ('x','y','traj')}},'test':{'x':d['x'][te]}}
    b=base.solve_batch(q); a=new._alternative(q)
    row=[]
    for w in (0.,.5,1.):
        y=(1-w)*b['y_hat']+w*a[0]; t=(1-w)*b['state_pred']+w*a[1]
        row.append([np.mean(abs(y-d['y'][te])),np.mean(abs(t-d['traj'][te]))])
    err.append(row); print('split; baseline / blend / expanded:',row)
e=np.array(err); weight=[]
for j in range(2):
    # Promote only if both the required slice and the worst alternate slice improve.
    valid=[i for i in (1,2) if e[0,i,j]<=e[0,0,j] and e[:,i,j].max()<e[:,0,j].max()-.001 and e[:,i,j].mean()<e[:,0,j].mean()]
    weight.append((min(valid,key=lambda i:e[:,i,j].max()) if valid else 0)*.5)
print('promoted weights',weight)
if any(weight):
    text=pathlib.Path('/tmp/expanded13.py').read_text().replace('def solve_batch(study):','def _baseline(study):')
    text+='''\n\ndef solve_batch(study):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):
        b=_baseline(study)
        a,t=_alternative(study)
        y=(1-%%Y%%)*b['y_hat']+%%Y%%*a
        state=(1-%%T%%)*b['state_pred']+%%T%%*t
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
'''.replace('%%Y%%',repr(weight[0])).replace('%%T%%',repr(weight[1]))
    pathlib.Path('/holdout_task/batch_solver.py').write_text(text)
