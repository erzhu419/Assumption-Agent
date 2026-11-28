import numpy as np, pickle, runpy, sys
from pathlib import Path
base=Path('/tmp/v02_final23.py').read_text().split('\ndef run_node')[0]; extra=Path('/tmp/augment23.py').read_text(); ns={};exec(base+extra,ns)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; i=np.arange(len(a['y'])); f=[runpy.run_path('/holdout_task/_frozen_nodes/node_%02d.py'%k)['run_node'] for k in (1,2)]
ys=[0.,.25,.5,.75,1.]; ts=[(t,r) for t in (0.,.25,.5,1.) for r in (0.,.02,.06,.15)]
def eval_split(tr,va):
    ss={**s,'train':{k:a[k][tr] for k in ('x','y','traj')},'test':{'x':a['x'][va]}}; d=ns['_predict23'](ss); q=f[0]({'__input__':ss})
    ey=np.array([abs(ns['_emit23'](d,y,0.,0.)['y_hat']-a['y'][va]).mean() for y in ys]); et=[]
    for t,r in ts:
        o=ns['_emit23'](d,0.,t,r); et.append(abs(f[1]({'node_00':o,'node_01':q})['state_pred']-a['traj'][va]).mean())
    return ey,np.array(et)
R=[]
for off in range(4):
    R.append(eval_split(i[i%4!=off],i[i%4==off])); print('stress',off,R[-1][0],R[-1][1],flush=True)
P=eval_split(i[:-20],i[-20:]); print('public diagnostics',P,flush=True)
opts=[]
for j in range(2):
    A=np.array([r[j] for r in R]); metrics=.65*A.mean(0)+.35*A.max(0)
    # Only consider gains on the independent folds; allow some fold variability.
    allowed=(metrics<=metrics[0]*1.015)&(P[j]<=P[j][0]+1e-7)
    order=np.where(allowed,metrics,np.inf); k=int(np.argmin(order)); opts.append(k); print('promotion',j,k,'mean',A[:,k].mean(),'worst',A[:,k].max(),'public',P[j][k],flush=True)
y=ys[opts[0]]; t,r=ts[opts[1]]
if any(opts):
    code=base+extra+'\n\ndef run_node(inputs: dict) -> dict:\n    return _emit23(_predict23(inputs["__input__"]), '+repr(y)+', '+repr(t)+', '+repr(r)+')\n'; Path('/holdout_task/node_00.py').write_text(code)
