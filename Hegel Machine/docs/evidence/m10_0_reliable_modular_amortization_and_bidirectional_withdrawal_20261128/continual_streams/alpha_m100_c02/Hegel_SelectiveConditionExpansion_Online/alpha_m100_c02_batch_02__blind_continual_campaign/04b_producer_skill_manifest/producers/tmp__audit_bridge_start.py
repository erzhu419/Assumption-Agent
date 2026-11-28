import pickle, sys
import numpy as np
sys.path.insert(0,'/holdout_task')
import upstream_transducer as u
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']
x,y=c['true_channel_loads'],c['bridge_raw']; z=u._basis(x)
axis=np.count_nonzero(x,axis=1)==1
idx=np.arange(len(x)); mix=idx[~axis]
outer=[(f'axis {j+1}',idx[axis & (np.argmax(abs(x),axis=1)==j)]) for j in range(4)]
outer += [(f'face {s:+d}/{j+1}',mix[s*x[mix,j]>=np.quantile(s*x[mix,j],.8)]) for j in range(4) for s in (-1,1)]
spikes=np.flatnonzero(np.linalg.norm(z@u._fit_bridge(x,y)-y,axis=1)>20)

def fit(a,b,method):
    def solve(w):
        return np.linalg.lstsq(a*w[:,None],b*w[:,None],rcond=None)[0]
    w=np.ones(len(a)); floor=1e-12*max(1,np.median(abs(b)))
    if method=='l1':
        # Convex row-norm loss first, before reducing the influence of spikes.
        for _ in range(32):
            coef=solve(w); e=np.linalg.norm(a@coef-b,axis=1)
            w=1/np.sqrt(np.hypot(e,floor))
    elif method=='radial':
        size=np.linalg.norm(b-np.median(b,axis=0),axis=1)
        w=np.minimum(1,(3*np.median(size)/np.maximum(size,floor))**2)
    for _ in range(16):
        coef=solve(w); e=np.linalg.norm(a@coef-b,axis=1)
        w=1/np.hypot(1,e/max(.3*np.median(e),floor))
    good=e<=max(6*np.median(e),floor)
    return np.linalg.lstsq(a[good],b[good],rcond=None)[0],good

print('spike rows and raw norms',list(zip(spikes,np.linalg.norm(y[spikes],axis=1))))
worst=np.zeros((3,2)); methods=('baseline','l1','radial')
for name,te in outer:
    tr=np.setdiff1d(idx,te); te=np.setdiff1d(te,spikes)
    expected=~np.isin(tr,spikes)
    vals=[]
    for k,method in enumerate(methods):
        coef,good=fit(z[tr],y[tr],method)
        fw=np.linalg.norm(z[te]@coef-y[te],axis=1).max()
        inv=np.linalg.norm(u._invert(y[te],coef)-x[te],axis=1).max() if fw<.02 else np.inf
        worst[k]=np.maximum(worst[k],(fw,inv))
        vals.append(f'{fw:.6g}/{inv:.6g} rejected={tr[~good].tolist()}')
        if fw>1: print(method,'clean design cond',np.linalg.cond(z[tr][expected]),'fit rank',np.linalg.matrix_rank(z[tr][good]))
    print(name,' | '.join(vals))
print('worst forward/inverse by method',dict(zip(methods,worst.tolist())))
