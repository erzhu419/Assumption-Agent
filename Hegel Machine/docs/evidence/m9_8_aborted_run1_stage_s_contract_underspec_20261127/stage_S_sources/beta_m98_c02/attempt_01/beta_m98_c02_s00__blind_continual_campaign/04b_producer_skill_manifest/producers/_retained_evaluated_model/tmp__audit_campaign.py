import importlib.util, pickle
from itertools import combinations, combinations_with_replacement
import numpy as np
spec = importlib.util.spec_from_file_location('batch', '/holdout_task/batch_solver.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('study fields', list(s))
c=s['sensor_calibration_sweeps']; x=np.asarray(c['ref_rosette']); y=np.asarray(c['raw_rosette']); bank=c['single_bank_sweeps']; print('sweep count',len(x),'batch, relay and state-like fields',[k for k in s if any(t in k.lower() for t in ('batch','relay','state','apparatus','bank','lineage'))]);
def d(x, degree):
    terms=[()] + [t for i in range(1,degree+1) for t in combinations_with_replacement(range(4),i)]
    return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
q=np.column_stack([x[:,i] for i in range(4)]+[(x[:,i]+sg*x[:,j])/np.sqrt(2) for i,j in combinations(range(4),2) for sg in (-1,1)])
models={'affine':d(x,1),'quadratic':d(x,2),'cubic':d(x,3)}
# Search rational axes and pairwise sums/differences using the inner folds only.
rat=[(f'axis_or_pair={i},a={a}', d(x,1)/(1+a*q[:,i,None]**2)) for i in range(q.shape[1]) for a in (.002,.01,.05,.2)]
def err(D,t,v):
    return D[v]@np.linalg.lstsq(D[t],y[t],rcond=None)[0]-y[v]
rng=np.random.default_rng(311); folds=np.empty(len(x),int); folds[rng.permutation(len(x))]=np.arange(len(x))%5
out={k:[] for k in (*models,'rational_nested')}
for i in range(5):
    t=folds!=i; v=~t
    for k,D in models.items(): out[k].append(err(D,t,v))
    name,D=min(rat,key=lambda item:sum(np.sum(err(item[1],t&(folds!=j),folds==j)**2) for j in range(5) if j!=i))
    out['rational_nested'].append(err(D,t,v))
for k,e in out.items():
    print('calibration',k,'worst fold/channel RMSE',max(np.sqrt(np.mean(a*a,axis=0)).max() for a in e),'pooled',np.sqrt(np.mean(np.concatenate(e)**2)))
D=d(x,1); L=np.linalg.lstsq(D,y,rcond=None)[0]; bi=bank['bank_id']; br=bank['ref_signal']; bankerr=L[0,bi]+br*L[1+bi,bi]-bank['raw_readout']; print('single-bank external worst bank RMSE',max(np.sqrt(np.mean(bankerr[bi==i]**2)) for i in range(4)))
