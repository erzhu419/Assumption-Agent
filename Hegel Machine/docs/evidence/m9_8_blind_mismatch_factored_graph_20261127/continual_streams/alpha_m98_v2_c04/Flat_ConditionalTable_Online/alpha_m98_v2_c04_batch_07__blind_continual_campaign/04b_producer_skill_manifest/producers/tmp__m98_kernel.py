import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/m98_arrays.npz'); globals().update({k:z[k] for k in z.files}); drift=th[2]+x@th[3:]; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
def maps(p,mode):
    T=tau.copy(); H=tau.copy(); a=np.ones(2); d=1.; L=0.
    if 'T' in mode: T*=np.exp(p[:2]); p=p[2:]
    if 'H' in mode: H*=np.exp(p[:2]); p=p[2:]
    if 'A' in mode: a+=p[:2]; p=p[2:]
    if 'L' in mode: L=p[0]; p=p[1:]
    G=(hold[:,None]-past@(-np.expm1(-dt[:,None]/H)*np.exp(-later[:,None]/H)))*th[:2]*a
    if mode[0]=='b': f,r=t,1.
    else:
        d=np.exp(p[0]); s=np.exp(p[1]);
        if mode[0]=='e': f,r=-d*s*np.expm1(-t/s),d
        if mode[0]=='r': f,r=d*t/(1+t/s),d
        if mode[0]=='g': f,r=d*s*np.log1p(t/s),d
        if mode[0]=='q': f,r=d*t/(1+(t/s)**2),d
    return G@(-np.expm1(-t[None,:]/T[:,None]))+drift[:,None]*(f+L*t),G@(1/T)+drift*(r+L)
def fit(mode,m):
    N=sum(2*int(k in mode) for k in 'THA')+('L' in mode)+(0 if mode[0]=='b' else 2)
    p=np.zeros(N)
    def err(p):
        c,r=maps(p,mode); return np.r_[((c[m]-yc[m])/.02).ravel(),(r[m]-yr[m])/.05]
    return least_squares(err,p,loss='soft_l1',f_scale=1.5,bounds=(-4.,4.),max_nfev=300).x if N else p
for mode in ['b','e','r','g','q','eL','eA','eT','eH','eTH','eATH','eATHL','rATHL','gATHL','qATHL']:
    ec=np.zeros_like(yc); er=np.zeros_like(yr)
    for g in np.unique(labs):
        m=labs==g; c,r=maps(fit(mode,~m),mode); ec[m]=(c-yc)[m]; er[m]=(r-yr)[m]
    p=fit(mode,np.ones(len(labs),bool)); c,r=maps(p,mode);
    def metric(c,r): return np.round([np.median(abs(c)),np.median(abs(r)),max(np.median(abs(c[labs==g])) for g in np.unique(labs)),max(np.median(abs(r[labs==g])) for g in np.unique(labs)),max(np.median(abs(c),axis=0))],5)
    print(mode,'OOF',metric(ec,er),'fit',metric(c-yc,r-yr),'p',p.round(5))
    if mode in ('e','eTH','eATHL'): print(' temporal signed median',np.median(c-yc,axis=0).round(4))
