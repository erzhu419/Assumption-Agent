exec(open('/tmp/alpha02_cv.py').read().split("for loss in ('cauchy'")[0])
for complete_fit in (False, True):
    e=np.empty_like(y); f=np.empty_like(r); params=[]
    for j in range(4):
        test=lab==j
        fit=~test & (~missing if complete_fit else True)
        p,w=m._fit(tuple(a[fit] for a in state),dt,time,y[fit],r[fit])
        A,B=m._design(p,tuple(a[test] for a in state),dt,time)
        e[test]=A@w-y[test]; f[test]=B@w-r[test]
        params.append(np.r_[np.exp(p),w])
    print('complete_fit',complete_fit,'fold parameters',np.round(params,5))
    for name,mask in [('complete',~missing),('missing',missing)]:
        slices=[mask & (lab==j) for j in range(4)]
        print(name,'pooled curve/rate median and p90',q(e[mask])[[0,2]],q(f[mask])[[0,2]],'worst lab-horizon strain median',round(max(np.median(abs(e[s]),axis=0).max() for s in slices),5),'worst lab rate median',round(max(np.median(abs(f[s])) for s in slices),5))
    fit=~missing if complete_fit else np.ones(n,dtype=bool)
    p,w=m._fit(tuple(a[fit] for a in state),dt,time,y[fit],r[fit]); print('full parameters',np.round(np.r_[np.exp(p),w],6))
