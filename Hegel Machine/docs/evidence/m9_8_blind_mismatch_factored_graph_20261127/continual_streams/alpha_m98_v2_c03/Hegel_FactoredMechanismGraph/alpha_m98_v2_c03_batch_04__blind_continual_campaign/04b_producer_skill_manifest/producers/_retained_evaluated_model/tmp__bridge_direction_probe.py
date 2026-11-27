import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
from upstream_transducer import _basis, _fit_bridge, _inverse
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; X=C['true_channel_loads']; Y=C['bridge_raw']; A=_basis(X)
mixed=np.count_nonzero(X,axis=1)>1
te=mixed & (X[:,3]>.7); tr=~te

def seed(a,y):
    rng=np.random.default_rng(21); p=a.shape[1]
    def score(c): return np.median(np.linalg.norm(y-a@c,axis=1))
    best=np.linalg.lstsq(a,y,rcond=None)[0]; loss=score(best)
    for _ in range(96):
        ids=rng.choice(len(a),min(len(a),p+5),replace=False)
        c, _, rank, _=np.linalg.lstsq(a[ids],y[ids],rcond=None)
        if rank!=p: continue
        s=score(c)
        if s<loss: best,loss=c,s
    return best

reference = tr & ~np.isin(np.arange(len(X)), [13, 14, 15])
for label,b in [('current', _fit_bridge(X[tr], Y[tr])), ('clean reference', _fit_bridge(X[reference], Y[reference])), ('subset initialization', seed(A[tr], Y[tr]))]:
    r=np.linalg.norm(Y-A@b,axis=1)
    print(label,'training residual median/max',np.median(r[tr]),max(r[tr]),'held-out median/max',np.median(r[te]),max(r[te]),'training largest',sorted(zip(r[tr],np.flatnonzero(tr)),reverse=True)[:7])
    if label=='subset initialization':
        keep=tr & (r<=6*np.median(r[tr])); b=_fit_bridge(X[keep],Y[keep]); print('subset-refit rejected training rows',np.flatnonzero(tr & ~keep).tolist())
    print(label,'inverse max error',np.max(np.abs(_inverse(Y[te],b)-X[te])))
print('design condition number (entire / directional training / directional training minus known spikes):',np.linalg.cond(A),np.linalg.cond(A[tr]),np.linalg.cond(A[tr & ~np.isin(np.arange(len(X)),[13,14,15])]))
