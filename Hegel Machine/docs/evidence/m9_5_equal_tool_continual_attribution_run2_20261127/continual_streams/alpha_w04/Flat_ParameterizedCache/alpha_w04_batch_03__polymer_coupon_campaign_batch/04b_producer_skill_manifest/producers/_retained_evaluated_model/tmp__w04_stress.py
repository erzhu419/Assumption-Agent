import importlib.util
import pickle
from pathlib import Path
import json
import numpy as np
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('coupon', root/'alpha_w04_batch_03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
p = pickle.load((root/'public_alpha_w04_batch_03.pkl').open('rb'))[0]
d, g = p['dataset'], p['gt']
print('split fields', {k:list(d[k]) for k in ('train','test')})
for lab in range(4):
    idx = d['train']['lab_id'] != lab
    tr = {k:v[idx] for k,v in d['train'].items()}
    o = s.solve_alpha_w04_batch_03(dict(d, train=tr))
    own = d['test']['lab_id'] == lab
    missing = np.isnan(d['test']['coupon_features_raw']).any(axis=1)
    for label, keep in [('all',own),('complete',own & ~missing),('imputed',own & missing)]:
        if keep.any():
            c = o['predicted_strain'][keep]-g['te_clean'][keep]
            r = o['initial_rate_pred'][keep]-g['te_true_rate0'][keep]
            print('leave-lab-out',lab,label,int(keep.sum()),'strain RMSE',np.sqrt(np.mean(c*c)),'rate RMSE',np.sqrt(np.mean(r*r)))
    print('tau rel max',np.max(abs(o['estimated_taus']/g['taus']-1)))
# Use real public histories with changed amplitudes to activate hold and history independently.
rng=np.random.default_rng(93)
taus=np.array([.59,3.94]); weights=np.array([.63,.54,.08,.15,-.13,.11,-.09,.21,.34]); yld=d['yield_stress']
cal=np.array([[1.22,.14],[.71,-.089],[1.20,.085],[.73,-.14]])
dt,times=d['past_dt'],d['eval_times']
xs=s._align(d['train'],d['test'])
def make(split, x):
    z=dict(split)
    n=len(x); j=np.arange(n)%4
    hist=z['past_stress_cmd']*np.where((j&1)[:,None]>0,2.5,0.75)
    hold=z['stress_hold_cmd'].copy(); hold[j>=2]=rng.uniform(3.2,4.7,(j>=2).sum())*np.where(hold[j>=2]>=0,1,-1)
    a,b=cal[z['lab_id']].T
    z['past_stress_cmd']=(hist-b[:,None])/a[:,None]; z['stress_hold_cmd']=(hold-b)/a
    slopes=weights[2]+x@weights[3:7]
    amplitudes=[]
    for tau in taus:
        current=np.zeros(n)
        for h,step in zip(hist.T,dt):
            current=h+(current-h)*np.exp(-step/tau)
        amplitudes.append(hold-current)
    amplitudes=np.array(amplitudes).T*weights[:2]
    ph=np.maximum(abs(hold)-yld,0)**2*np.sign(hold)
    work=np.sum(np.maximum(abs(hist)-yld,0)**2*dt,axis=1)*hold
    z['strain_curves_obs']=z['strain_0'][:,None]+(amplitudes@(1-np.exp(-times/taus[:,None])))+(slopes+weights[7]*ph)[:,None]*times+weights[8]*work[:,None]*(1-np.exp(-times/taus[0]))
    z['instant_rate_obs']=amplitudes@(1/taus)+slopes+weights[7]*ph+weights[8]*work/taus[0]
    return z,(ph!=0).astype(int)+2*(work!=0)
tr,_=make(d['train'],xs[0]); te,reg=make(d['test'],xs[1]); labs=np.repeat(np.arange(4),8); x=np.tile(np.linspace(-2,2,8),4); y=cal[labs,0]*x+cal[labs,1]
y[[1,14,18,27]] += [.5,-.4,.7,-.5]
out=s.solve_alpha_w04_batch_03(dict(d,train=tr,test=te,calibration_load_cells={'lab_id':labs,'cmd_stress':x,'true_stress':y}))
print('mixed-regime parameters',out['estimated_taus'],out['viscoplastic_weights'])
assert np.max(abs(out['estimated_taus']-taus))<1e-5
assert np.max(abs(np.r_[out['hereditary_weights'],out['viscoplastic_weights']]-weights))<1e-5
for label in range(4):
    m=reg==label; assert m.any()
    e=np.max(abs(out['predicted_strain'][m]-te['strain_curves_obs'][m])); r=np.max(abs(out['initial_rate_pred'][m]-te['instant_rate_obs'][m])); print('mixed regime',label,int(m.sum()),e,r); assert max(e,r)<1e-5
# Retain three or fewer checks in each historical regime and inject one EMI glitch.
history=json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
for panel in history:
    labs=np.asarray(panel['lab_id']); x=np.asarray(panel['cmd_stress']); y=np.asarray(panel['true_stress']); want=np.array([np.polyfit(x[labs==l],y[labs==l],1) for l in range(4)])
    ids=np.concatenate([np.flatnonzero(labs==l)[np.argsort(x[labs==l])[[0,4,9]]] for l in range(4)])
    noisy=y[ids].copy(); noisy[1::3] += .5
    got=s._calibrate({'lab_id':labs[ids],'cmd_stress':x[ids],'true_stress':noisy})
    e=np.max(abs(got-want)); print('sparse EMI',panel['episode_short_name'],e); assert e < .001
