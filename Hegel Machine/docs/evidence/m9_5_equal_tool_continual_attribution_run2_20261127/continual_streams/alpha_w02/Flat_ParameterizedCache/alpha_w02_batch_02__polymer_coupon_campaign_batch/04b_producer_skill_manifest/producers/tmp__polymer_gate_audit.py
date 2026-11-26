import pickle, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, '/holdout_task')
import alpha_w02_batch_02_solver as m
with open('/holdout_task/public_alpha_w02_batch_02.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
t = d['train']; n = len(t['lab_id']); labs = t['lab_id']; st = t['stratum']
a = m._inputs(d); tr = tuple(v[:n] for v in a); dt, ts = d['past_dt'], d['eval_times']
y = np.c_[t['strain_curves_obs']-t['strain_0'][:,None], t['instant_rate_obs']]
missing = np.isnan(t['coupon_features_raw']).any(axis=1)
original = m._components
# Pairwise linear combinations are already in the affine span; check nonlinear
# stress and feature couplings, and a fixed saturating stress denominator.
def expanded(inputs, dt, tau, variant):
    delta, z = original(inputs, dt, tau)
    past, hold, mem, feat = inputs
    if variant == 1:
        i,j = np.triu_indices(feat.shape[1]); z = np.c_[z, feat[:,i]*feat[:,j], hold[:,None]*z]
    if variant == 2:
        q = hold/d['yield_stress']; z = np.c_[z, z*(q*q/(1+q*q))[:,None]]
    return delta,z
m._components = expanded
for v in range(3):
    r = np.empty_like(y)
    for lab in np.unique(labs):
        test = labs==lab; train = ~test
        tau,c = m._fit(tuple(w[train] for w in tr),dt,ts,y[train],v)
        r[test] = m._design(tuple(w[test] for w in tr),dt,ts,tau,v)@c-y[test]
    r[missing,:-1] -= r[missing,-1,None]*ts
    print('basis',v,'worst lab complete/missing median curve error',*[round(max(np.median(abs(r[(labs==lab)&(missing==flag),:-1])) for lab in np.unique(labs)),6) for flag in (False,True)],'worst complete lab endpoint median AE', round(max(np.median(abs(r[(labs==lab)&~missing,-2])) for lab in np.unique(labs)),6))
    if v==0:
        print('complete-case rate residual signed median by lab x stratum',np.array([[np.median(r[(labs==lab)&(st==s)&~missing,-1]) for s in np.unique(st)] for lab in np.unique(labs)]).round(5))
m._components = original
u=d['test']; tau,c=m._fit(tr,dt,ts,y); test=tuple(w[n:] for w in a)
delta,z=m._components(test,dt,tau); amp=delta*c[:2]; drift=z@c[2:]; miss=np.isnan(u['coupon_features_raw']).any(axis=1); drift[miss]=u['instant_rate_obs'][miss]-amp[miss]@(1/tau)
# Check first crossing against a fine bracketing grid on fitted real coupons,
# including thresholds just below an interior maximum when one is present.
grid=np.linspace(0,ts[-1],20001); initial=u['strain_0']; threshold=u['critical_strain'].copy(); errors=[]; turning=0; pulses=0
for pass_id in range(2):
    pred=m._hazard(initial,threshold,ts,tau,amp,drift)
    for i in range(len(initial)):
        def value(t): return initial[i]+(-np.expm1(-np.asarray(t)[...,None]/tau))@amp[i]+drift[i]*t
        wave=value(grid); j=np.argmax(wave)
        hits=np.flatnonzero(wave>=threshold[i]); ref=ts[-1] if not len(hits) else (0. if hits[0]==0 else brentq(lambda t:value(t)-threshold[i],grid[hits[0]-1],grid[hits[0]]))
        errors.append(abs(ref-pred[i]))
        if pass_id==0:
            turning+=int(np.any(np.diff(wave)[:-1]*np.diff(wave)[1:]<0))
            if 0<j<len(grid)-1 and wave[j]>max(wave[0],wave[-1]):
                threshold[i]=wave[j]-.01*(wave[j]-max(wave[0],wave[-1])); pulses+=1
print('real fitted turning trajectories',turning,'interior-only crossing stress cases',pulses,'max first-crossing discrepancy',max(errors)); assert max(errors)<1e-8
