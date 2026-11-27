import json, pickle
from itertools import combinations_with_replacement
import numpy as np

log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for entry in log:
    trace=entry['facility_telemetry_trace']
    print('apparatus',entry['completed_campaign_short_name'],type(trace).__name__,len(trace),str(trace)[:240],str(trace)[-460:])
ds=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
c=ds['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']
axis=np.count_nonzero(x,axis=1)==1
terms=[()] + [(j,) for j in range(4)] + list(combinations_with_replacement(range(4),2)) + list(combinations_with_replacement(range(4),3))
a=np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
mixed=np.flatnonzero(~axis); lone=np.flatnonzero(axis)
folds=[]
for j,block in enumerate(np.array_split(mixed[np.random.default_rng(31).permutation(len(mixed))],4)):
    folds.append(np.r_[block,lone[j::4]])
# Median-based residual scale and hard row rejection, refit independently per fold.
def fit(a,y):
    keep=np.ones(len(a),bool)
    for _ in range(15):
        b=np.linalg.lstsq(a[keep],y[keep],rcond=None)[0]
        d=np.linalg.norm(a@b-y,axis=1)
        nxt=d<=max(6*np.median(d),.00075)
        if nxt.sum()<a.shape[1] or np.array_equal(nxt,keep): break
        keep=nxt
    return b,keep
models={'affine':lambda t:len(t)<=1,'diagonal_cube':lambda t:len(t)<=1 or len(t)==3 and len(set(t))==1,'all_quadratic_plus_pure_cubes':lambda t:len(t)<=2 or len(set(t))==1,'full_cubic':lambda t:True}
for name,sel in models.items():
    idx=[i for i,t in enumerate(terms) if sel(t)]; z=a[:,idx]; pred=np.zeros_like(y)
    for f in folds:
        train=np.setdiff1d(np.arange(len(x)),f); b,k=fit(z[train],y[train]); pred[f]=z[f]@b
    dist=np.linalg.norm(pred-y,axis=1)
    print(name,'axis med/max',np.quantile(dist[axis],[.5,1]),'mixed med/q80/q90/max',np.quantile(dist[~axis],[.5,.8,.9,1]))
    b,k=fit(z,y)
    print('trimmed:',np.flatnonzero(~k),'full-fit largest five clean',np.sort(np.linalg.norm(z[k]@b-y[k],axis=1))[-5:])
    if name!='affine':
        for i,row in zip(idx,b):
            if len(terms[i])<2 or np.max(abs(row))>.001: print(terms[i],row.round(6))
