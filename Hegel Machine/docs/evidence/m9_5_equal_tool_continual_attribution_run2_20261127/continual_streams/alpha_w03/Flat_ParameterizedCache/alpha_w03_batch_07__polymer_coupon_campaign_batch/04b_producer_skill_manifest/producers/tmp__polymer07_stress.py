import runpy
import numpy as np
s = runpy.run_path('/tmp/polymer07_cv.py')
m,d,x,h,p,dt,t,q,z,rate,lab,complete = (s[k] for k in ('m','d','x','h','p','dt','t','q','z','rate','lab','complete'))
for j in range(4):
    tr, te = lab!=j,lab==j
    fit = tr & complete
    tau,w = m._fit(x[fit],h[fit],p[fit],dt,t,q,z[fit],rate[fit])
    C,R = m._design(tau,x[te],h[te],p[te],dt,t,q)
    innovation = rate[te]-R@w
    wing = m._wing_blend(x[tr],h[tr],p[tr],dt,t,q,z[tr],rate[tr],lab[tr],complete[tr])
    blend = np.where(complete[te],wing[j//2],1.)
    e = C@w + (blend*innovation)[:,None]*t-z[te]
    print('held-out lab',j,'curve RMSE',np.sqrt(np.mean(e**2)),'curve median abs',np.median(np.abs(e)),'onset RMSE',np.sqrt(np.mean(((1-blend)*innovation)**2)),'tau',tau)
# Recovery test uses measured features and histories, with independent above-yield excitation.
rng=np.random.default_rng(3)
known_tau=np.array([.46,3.2]); known_coef=np.array([.7,.55,.08,.15,-.13,.11,-.09,.21,.17])
hh=h.copy(); pp=p.copy()
hh[::3]=rng.choice([-1,1],len(hh[::3]))*rng.uniform(q+.15,q+1.4,len(hh[::3]))
pp[1::3,-3]=rng.choice([-1,1],len(pp[1::3]))*rng.uniform(q+.15,q+1.4,len(pp[1::3]))
for a,b in [(False,False),(True,False),(False,True),(True,True)]:
    hc,pc=(hh if a else h),(pp if b else p)
    expected=known_coef.copy(); expected[7:]*=[a,b]
    C,R=m._design(known_tau,x,hc,pc,dt,t,q)
    tau,w=m._fit(x,hc,pc,dt,t,q,C@known_coef,R@known_coef)
    e=max(np.max(np.abs(tau-known_tau)),np.max(np.abs(w-expected)))
    print('hold/history activated',a,b,'max parameter error',e,'plastic',w[7:]); assert e < 1e-5
