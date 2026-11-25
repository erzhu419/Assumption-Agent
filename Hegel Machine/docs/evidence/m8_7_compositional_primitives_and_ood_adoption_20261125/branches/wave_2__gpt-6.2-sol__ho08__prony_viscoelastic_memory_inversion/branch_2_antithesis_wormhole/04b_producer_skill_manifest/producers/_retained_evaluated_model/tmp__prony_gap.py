import numpy as np
from prony_rheology import _spectrum, _basis, _history
from pathlib import Path
# Reuse the independent reference and test protocols, not the validation loop.
exec(Path('/tmp/prony_validate.py').read_text().split('for i, (G,tau,amp) in enumerate(regimes):')[0])
for i, (G, tau, amp) in enumerate(regimes[1:], 1):
    T = tj if i%2 else t
    clean = reference(T, e, G, tau, amp)
    z = reference(te, et, G, tau, amp)
    step = np.zeros(te.size, bool)
    for u in (8,16): step |= (te >= u) & (te < u+.35)
    for seed in (0,1,2,3):
        r = np.random.default_rng(6000+seed)
        y = clean + (-2.4+.17*T if seed%2 else 1.3-.22*T) + r.normal(0, .015, T.size)
        ix = r.choice(T.size, round(.02*T.size), replace=False)
        y[ix] += r.choice([-1,1], size=ix.size) * (4.5 if seed%2 else r.uniform(.6,7,ix.size))
        for pen in (0.,.5):
            c,q = _spectrum(T,e,y,penalty=pen)
            order=np.argsort(q)
            err=_basis(_history(te,et),q)@c-z
            print('true',tau,'seed',seed,'pen',pen,'fit',q[order].round(4), 'amp',c[1:][order].round(2),'Gerr',round(c[0]/G-1,4),'stress',round(np.linalg.norm(err)/np.linalg.norm(z),4),'step share',round(np.linalg.norm(err[step])/max(np.linalg.norm(err),1e-20),3),flush=True)
