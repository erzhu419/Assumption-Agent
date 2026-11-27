import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _basis, _fit_bridge, _forward, _invert
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl','rb') as f:
    items = pickle.load(f)
for item in items:
    ds = item['dataset']; c = ds['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']
    print('dataset keys:', sorted(ds), 'chamber trial type:', type(ds.get('chamber_verification_trials')).__name__)
    a = _basis(x)
    i = np.arange(len(x)); axis = (x != 0).sum(1) == 1
    mixed = ~axis
    edge = np.max(abs(x), axis=1) >= np.quantile(np.max(abs(x[mixed]), axis=1), .75)
    folds = [(f'axis-{j}', axis & (np.argmax(abs(x),axis=1)==j)) for j in range(4)]
    folds += [('mixed-edge', mixed & edge)]
    folds += [(f'mixed-{j}', mixed & ~edge & (i%4==j)) for j in range(4)]
    def shared(a,y):
        coef = _fit_bridge(a[:, :, 1], y)
        keep = np.linalg.norm(np.einsum('nck,ck->nc',a,coef)-y,axis=1) < .01
        a, y = a[keep], y[keep]
        ratio = np.mean(coef[:, 2:]/coef[:, 1:2],axis=0)
        from scipy.optimize import least_squares
        def calc(r):
            v = a[:, :, 1] + a[:, :, 2:] @ r
            vc = v - v.mean(0); yc = y - y.mean(0)
            gain = np.sum(vc * yc,axis=0)/np.sum(vc**2,axis=0)
            offset = y.mean(0)-v.mean(0)*gain
            return offset, gain, offset + gain*v - y
        r = least_squares(lambda r: calc(r)[2].ravel(), ratio).x
        off, gain, _ = calc(r)
        return np.c_[off, gain, gain[:,None]*r]
    self_, nxt, prev = (a[:,:,k] for k in (1,2,3))
    # Linear pairwise sums/differences are already spanned by the linear basis.
    # Distinct interactions and saturating contrasts can show up in residuals.
    aug = np.concatenate((a, np.stack([self_*prev,nxt*prev,self_**2,nxt**2,
        self_**2*nxt,self_*nxt**2,(self_-nxt)/(1+(self_-nxt)**2),
        (self_+nxt)/(1+(self_+nxt)**2),self_**3/(1+self_**2),
        self_*nxt/(1+(self_-nxt)**2)],axis=-1)),axis=-1)
    stats=[]
    for name, te in folds:
        tr = ~te; b = _fit_bridge(x[tr], y[tr]); s = shared(a[tr],y[tr])
        mask = tr & (np.linalg.norm(_forward(x,b)-y,axis=1)<.01)
        good = te & (np.linalg.norm(_forward(x,b)-y,axis=1)<.01)
        z = np.stack([np.linalg.lstsq(aug[mask,j],y[mask,j],rcond=None)[0] for j in range(4)])
        p = [_forward(x[good],b), _forward(x[good],s), np.einsum('nck,ck->nc',aug[good],z)]
        e = np.array(p)-y[good]
        inv = [np.max(abs(_invert(y[good],k)-x[good])) for k in (b,s)]
        stats.append((name,e.reshape(3,-1),inv))
        print(name, 'retained/spikes', int(good.sum()),int((te & ~good).sum()), 'raw rmse base/shared/aug', np.sqrt((e*e).mean((1,2))).round(8), 'load max base/shared',np.round(inv,8))
    print('pooled raw rmse',np.sqrt(np.mean(np.concatenate([e for _,e,_ in stats],axis=1)**2,axis=1)), 'worst raw error',np.max(np.abs(np.concatenate([e for _,e,_ in stats],axis=1)),axis=1),'worst inverse',np.max([v for _,_,v in stats],axis=0))
    print('operating-range nonlinear inverse residuals:')
    b = _fit_bridge(x,y)
    for name in ('train','test','chamber_verification_trials'):
        if ds.get(name) is not None:
            for k in ('past_channel_raw','hold_channel_raw'):
                r = ds[name][k]; v = _invert(r,b).reshape(-1,4)
                delta = np.max(abs(_forward(v,b)-r.reshape(-1,4)))
                assert delta < 1e-10
                print(name,k,len(v),f'{delta:.3g}')
    for k,v in solve_batch(ds).items():
        assert np.isfinite(v).all(), k
