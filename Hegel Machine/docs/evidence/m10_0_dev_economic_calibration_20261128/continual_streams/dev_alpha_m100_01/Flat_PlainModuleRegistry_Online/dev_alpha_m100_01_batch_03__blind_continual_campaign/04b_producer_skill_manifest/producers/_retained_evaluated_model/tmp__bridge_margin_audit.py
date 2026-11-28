import json, pickle, sys
from itertools import combinations
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
with open('public_batch.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
c = ds['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']
s = c['single_channel_sweeps']; ids = np.arange(len(x))
print('sections', {k:list(ds[k]) for k in ('train','test')})
print('calibration metadata', list(c), 'sweep metadata', list(s))
log = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for rec in log:
    print('apparatus', str(rec)[:850])
b = u._fit_bridge(x,y,s)
print('full coefficients', np.round(b,7).tolist())
print('cross/diagonal normalized', np.round(b[1:] / np.diag(b[1:5]),6).tolist())

def features(a, j, option):
    z = u._design(a,j); own = a[:,j]; nxt = a[:,(j+1)%4]
    if option == 'opposite-zero':
        return np.delete(z, 1+(j+2)%4, axis=1)
    if option == 'pairwise':
        return np.column_stack((z, own**2, nxt**2, own**2*nxt, own*nxt**2, nxt**3))
    if option.startswith('quotient'):
        strength = float(option.split(':')[1]); q = own if option.endswith('own') else own-nxt
        return np.column_stack((z[:,:5], z[:,5:] / (1 + strength*q[:,None]**2)))
    return z

def make_fit(train, option):
    return [u._fit_channel(features(x[train],j,option),y[train,j]) for j in range(4)]

def prediction(test, option, fit):
    return np.column_stack([features(x[test],j,option) @ fit[j] for j in range(4)])

# Identical clean evaluation rows for all challengers; the previous independent folds identified these gross spikes.
clean = ~np.isin(ids,[13,14,15]); radius = np.max(abs(x),axis=1)
folds = {'random':np.array_split(np.random.default_rng(31).permutation(ids),5), 'axis':[np.arange(i,i+4) for i in range(0,16,4)], 'outer':[ids[radius>=np.quantile(radius,.8)]]}
linear = b[1:5].T
for name, tests in folds.items():
    for family in ('cyclic','opposite-zero','pairwise','quotient'):
        errors=[]; total=[]; chosen=[]
        for ts in tests:
            tr=np.setdiff1d(ids,ts); te=ts[clean[ts]]
            option=family
            if family=='quotient':
                options=['cyclic']+[f'quotient:{a}:{q}' for a in (.001,.01,.1,1) for q in ('own','difference')]
                inner=np.array_split(np.random.default_rng(37).permutation(tr),3)
                def score(o):
                    e=[]
                    for v in inner:
                        f=make_fit(np.setdiff1d(tr,v),o); ev=prediction(v,o,f)-y[v]
                        e.extend(np.linalg.norm(ev,axis=1))
                    return np.quantile(e,.8)
                option=min(options,key=score)
            fit=make_fit(tr,option); top=prediction(te,option,fit)
            err=top-y[te]; errors.extend(np.linalg.norm(err,axis=1))
            # Translate raw errors into the downstream-relevant total-load direction
            # using the local Jacobian of the baseline trained on this outer fold.
            base=np.column_stack(make_fit(tr,'cyclic')); z=x[te]
            jac=np.broadcast_to(base[1:5].T,(len(te),4,4)).copy()
            j=u._CHANNELS; n=u._NEXT
            jac[:,j,j]+=base[5]*z[:,n]+3*base[6]*z**2; jac[:,j,n]+=base[5]*z
            total.extend(abs(np.linalg.solve(jac,err[...,None])[...,0].sum(axis=1)))
            chosen.append(option)
        print(name,family,'raw p90/max',np.quantile(errors,.9),max(errors),'local total-load max',max(total),'options',','.join(chosen))
