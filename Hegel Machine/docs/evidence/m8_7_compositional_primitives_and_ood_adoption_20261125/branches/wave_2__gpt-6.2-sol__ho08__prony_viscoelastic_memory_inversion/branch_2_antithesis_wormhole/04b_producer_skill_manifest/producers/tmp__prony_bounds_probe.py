from pathlib import Path
exec(Path('/tmp/prony_validate.py').read_text().split('for i, (G,tau,amp) in enumerate(regimes):')[0])
from prony_rheology import _pilot
for i in (1,2,4,5):
    G,tau,amp=regimes[i]; T=tj if i%2 else t
    clean=reference(T,e,G,tau,amp); z=reference(te,et,G,tau,amp)
    H=_history(T,e); HT=_history(te,et)
    D=np.c_[np.ones(T.size),T/T[-1]-.5]
    for seed in range(8):
        r=np.random.default_rng(6000+seed)
        y=clean+(-2.4+.17*T if seed%2 else 1.3-.22*T)+r.normal(0,.015,T.size)
        ix=r.choice(T.size,round(.02*T.size),replace=False)
        y[ix]+=r.choice([-1,1],size=ix.size)*(4.5 if seed%2 else r.uniform(.6,7,ix.size))
        c,q=_spectrum(T,e,y); err=metric(_basis(HT,q)@c,z)
        if err>.022 or abs(c[0]/G-1)>.15:
            _,rr,s=_pilot(_basis(H,np.geomspace(.02,20,80)),D,y)
            j=abs(rr)>6*s; sort=np.argsort(q)
            print(i,seed,'errors',np.round([err,c[0]/G-1],4),'tau',q[sort].round(4),'amp',c[1:][sort].round(2),'miss/extra',sum(~j[ix]),sum(j)-sum(j[ix]),flush=True)
