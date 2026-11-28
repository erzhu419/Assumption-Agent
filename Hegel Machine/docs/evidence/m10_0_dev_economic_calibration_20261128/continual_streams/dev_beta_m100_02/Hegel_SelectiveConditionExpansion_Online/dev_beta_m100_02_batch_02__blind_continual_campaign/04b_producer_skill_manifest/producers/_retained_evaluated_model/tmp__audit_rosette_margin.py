import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
sweep = study['sensor_calibration_sweeps']
x, y = sweep['ref_rosette'], sweep['raw_rosette']
bank = sweep['single_bank_sweeps']
rows = np.arange(len(x))
brows = np.arange(len(bank['bank_id']))
clean = ~np.isin(rows, [13, 14, 15])

# A rational challenger divides the nonlinear correction by 1+a*q^2,
# with q either the local signal or the adjacent-channel difference.
def design(z, j, a, difference):
    d = up._design(z, j)
    q = z[:, j] - z[:, (j+1)%4] if difference else z[:, j]
    d[:, 5:] /= 1 + a*q[:, None]**2
    return d

def fit(tr, bk, a, diff):
    if a == 0:
        return up._fit(x[tr], y[tr], {k:v[bk] for k,v in bank.items()})
    A = [design(x[tr], j, a, diff) for j in range(4)]
    B, v = [], []
    for j in range(4):
        ii = bk[bank['bank_id'][bk] == j]
        z = np.zeros((len(ii), 4)); z[:, j] = bank['ref_signal'][ii]
        B.append(design(z, j, a, diff)); v.append(bank['raw_readout'][ii])
    w = np.ones(len(tr)); wb = [np.ones(len(t)) for t in v]
    for _ in range(18):
        coef = np.column_stack([np.linalg.lstsq(np.vstack((d*w[:,None], b*u[:,None])), np.r_[y[tr,j]*w,t*u], rcond=None)[0] for j,(d,b,t,u) in enumerate(zip(A,B,v,wb))])
        e = np.linalg.norm(np.column_stack([d@coef[:,j] for j,d in enumerate(A)])-y[tr],axis=1)
        scale = max(np.median(e)/1.83,1e-8)
        w = 1/np.hypot(1,e/(4*scale))
        wb = [1/np.hypot(1,(b@coef[:,j]-t)/(3*scale)) for j,(b,t) in enumerate(zip(B,v))]
    return coef

def errors(test, p, a, diff):
    z = x[test]
    return np.linalg.norm(np.column_stack([design(z,j,a,diff)@p[:,j] for j in range(4)])-y[test], axis=1)

outer = {f'random{i}':t for i,t in enumerate(np.array_split(np.random.default_rng(31).permutation(rows),5))}
outer['upper envelope'] = np.argsort(np.max(abs(x),axis=1))[-16:]
outer['joint contrast'] = np.argsort(np.max(abs(x-np.roll(x,-1,axis=1)),axis=1))[-16:]
outer['central'] = np.argsort(np.linalg.norm(x,axis=1))[:16]
options = [(0.,False)] + [(a,d) for d in (False,True) for a in (1e-5,1e-4,1e-3,1e-2,0.1)]
base_max = selected_max = 0.
for i,(name,test) in enumerate(outer.items()):
    tr = np.setdiff1d(rows,test)
    bk = brows[brows%5 != i%5]
    def score(option):
        a,d = option
        parts = []
        for val in np.array_split(np.random.default_rng(92).permutation(tr),3):
            p = fit(np.setdiff1d(tr,val),bk,a,d)
            parts.extend(errors(val,p,a,d))
        return np.quantile(parts,0.9)
    a,d = min(options,key=score)
    test = test[clean[test]]
    p0 = fit(tr,bk,0.,False)
    baseline = max(errors(test,p0,0.,False))
    inverse = max(np.linalg.norm(up._inverse(y[test],p0)-x[test],axis=1))
    selected = max(errors(test,fit(tr,bk,a,d),a,d))
    base_max = max(base_max,baseline); selected_max = max(selected_max,selected)
    print(f'{name}: forward={baseline:.3e} inverse={inverse:.3e} nested-challenger={selected:.3e} a={a:g} difference={d}')
print(f'Worst forward: cyclic={base_max:.3e}; nested challenger={selected_max:.3e}')
