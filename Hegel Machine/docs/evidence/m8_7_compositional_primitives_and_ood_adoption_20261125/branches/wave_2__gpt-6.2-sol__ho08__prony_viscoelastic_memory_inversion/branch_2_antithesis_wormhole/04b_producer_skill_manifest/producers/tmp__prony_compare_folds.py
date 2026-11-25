import importlib.util
import os
from pathlib import Path
import pickle
import numpy as np

p = pickle.load(open('/holdout_task/public_rheology.pkl', 'rb'))
x = p['experiment']
t,e,y,tt,et = (x[k] for k in ('train_t','train_strain','train_stress_raw','test_t','test_strain'))
blocks = ((0,2),(2,6),(6,12),(12,18),(18,24),(24,31))

def load(path, name):
    text = Path(path).read_text().replace('penalty=.5):', 'penalty=.5, observed=None):', 1)
    text = text.replace('    c, residual, scale = _pilot(a, drift, y)', '''    if observed is None:
        observed = np.ones(len(t), dtype=bool)
    c, r, scale = _pilot(a[observed], drift[observed], y[observed])
    residual = np.zeros(len(t))
    residual[observed] = r''')
    text = text.replace('    good = ~spike', '    good = observed & ~spike')
    text = text.replace('    return coef, np.exp(logtau)\n', '    return coef, np.exp(logtau)\n')
    scope = {}
    exec(compile(text,path,'exec'),scope)
    return scope

for path, label, cost, soft in [('/holdout_task/prony_rheology.py','baseline',4,0),('/tmp/prony_joint/prony_rheology.py','joint hard',4,0),('/tmp/prony_joint/prony_rheology.py','joint average',4,1),('/tmp/prony_profile/prony_rheology.py','fast profile',4,1)]:
    os.environ['PRONY_ORDER_COST'] = str(cost)
    os.environ['PRONY_SOFT_ORDER'] = str(soft)
    src = load(path,label)
    fit,basis,history = (src[k] for k in ('_spectrum','_basis','_history'))
    c,tau = fit(t,e,y)
    z = basis(history(tt,et),tau)@c
    scores = []
    gs = []
    for lo,hi in blocks:
        held = (t>=lo)&(t<hi)
        c0,tau0 = fit(t,e,y,observed=~held)
        r = y - basis(history(t,e),tau0)@c0
        d = np.c_[np.ones(len(t)),t/t[-1]-.5]
        # Estimate the affine nuisance from training residuals only.
        dw, rw = d[~held], r[~held]
        b = np.linalg.lstsq(dw,rw,rcond=None)[0]
        for _ in range(12):
            err = rw-dw@b
            scale = src['_mad'](err)
            w = 1/np.hypot(1,err/(3*scale))
            b = np.linalg.lstsq(dw*w[:,None],rw*w,rcond=None)[0]
        # The public spikes are separated by several stress units; a cap fixed
        # from training alone retains all smaller held-out fit errors.
        err = (r-d@b)[held]
        keep = abs(err) < 15*scale
        scores.append(np.sqrt(np.mean(err[keep]**2)))
        gs.append(c0[0])
    print(label,'stress/G error',np.round([np.linalg.norm(z-p['gt_test_stress'])/np.linalg.norm(p['gt_test_stress']),abs(c[0]/p['gt_G_inf']-1)],6),'blocked',np.round(scores,5),'worst',round(max(scores),5),'blocked G',np.round(gs,3),flush=True)
