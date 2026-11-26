import itertools, sys, time
import numpy as np
sys.path.insert(0, '/holdout_task')
from coupled_interferometry_certified_l0 import solve_coupled_interferometry_l0, _selection


def verify(cert, x, y, ridge, l0, box):
    p = x.shape[1]
    w = np.asarray(cert['weights'])
    r = x @ w - y
    f = .5*(r@r + ridge*(w@w)) + l0*np.count_nonzero(abs(w)>1e-8)
    assert w.shape == (p,) and np.max(abs(w)) <= box and abs(f-cert['objective']) < 1e-9
    leaves = cert['subcube_leaves']
    assert 1 <= len(leaves) <= 2048, len(leaves)
    allbits = np.arange(1<<p, dtype=np.int32)
    covered = np.zeros(1<<p, dtype=np.int8)
    gram = x.T@x
    for leaf in leaves:
        z, k = leaf['fixed_zero'], leaf['fixed_active']
        assert len(z)==len(k)==p and all(type(v) is bool for v in z+k)
        assert not any(a and b for a,b in zip(z,k))
        zm = sum(1<<j for j,v in enumerate(z) if v)
        km = sum(1<<j for j,v in enumerate(k) if v)
        covered[(allbits & (zm|km)) == km] += 1
        d = np.asarray(leaf['minorant_diagonal']); c = np.asarray(leaf['minorant_center'])
        assert np.min(d) >= 0 and np.linalg.eigvalsh(gram-np.diag(d))[0] >= -1e-8
        g = gram@c - x.T@y
        b = g-d*c; a = d+ridge
        u = np.array([min(box, abs(bj)/aj) if aj>0 else box for aj,bj in zip(a,b)])
        opt = .5*a*u*u - abs(b)*u + l0
        base = .5*np.sum((x@c-y)**2) - g@c + .5*np.dot(d,c*c)
        check = base + sum(0 if zz else oo if kk else min(0,oo) for zz,kk,oo in zip(z,k,opt))
        assert leaf['lb'] <= check + 1e-8, (leaf['lb'],check)
    assert (covered==1).all()
    assert cert['certified_lower_bound'] <= min(q['lb'] for q in leaves)+1e-9
    gap = f-cert['certified_lower_bound']
    assert -1e-9 <= gap <= 1e-5, (gap, len(leaves))
    return len(leaves),gap


def oracle(x,y,ridge,l0,box):
    p=x.shape[1]; q=x.T@x+ridge*np.eye(p); b=x.T@y
    best=.5*(y@y)
    for face in itertools.product((0,1,-1,2),repeat=p):
        f=np.array(face); j=f==2; k=abs(f)==1
        w=np.zeros(p); w[k]=box*f[k]
        if np.any(j): w[j]=np.linalg.solve(q[np.ix_(j,j)],b[j]-q[np.ix_(j,k)]@w[k])
        if np.max(abs(w))>box+1e-10: continue
        r=x@w-y
        best=min(best,.5*(r@r+ridge*(w@w))+l0*np.count_nonzero(abs(w)>1e-8))
    return best

rng=np.random.default_rng(7319)
for seed in range(6):
    x=rng.normal(size=(26,6)); x[:,1]=.91*x[:,0]+.3*x[:,1]
    y=rng.normal(size=26)+x@rng.normal(size=6)
    ridge, l0, box=.04, .3, .18 + .22*seed
    cert=_selection(x,y,ridge,l0,box)
    leaves,gap=verify(cert,x,y,ridge,l0,box)
    err=abs(cert['objective']-oracle(x,y,ridge,l0,box))
    assert err < 2e-7,err
    print('oracle',seed,leaves,gap,err,flush=True)
for p in (18,19):
    for spread in (.12,.72,1.35):
        n=1800; nc=1000; ns=115
        center=np.resize(np.arange(5),n); rng.shuffle(center)
        s=(rng.random(n)<(.23+.10*center)).astype(int)+(rng.random(n)<.22)
        z=rng.normal(size=(n,8)) + .25*s[:,None]; gains=np.exp(rng.uniform(-.65,.65,5)); shifts=rng.uniform(-1.3,1.3,5)
        control=rng.normal(size=(n,18))*gains[center,None]+shifts[center,None]
        raw=z*gains[center,None]+shifts[center,None]
        control[rng.random(control.shape)<.05]=np.nan; raw[rng.random(raw.shape)<.025]=np.nan
        phase=rng.uniform(-2.2,2.2,(n,2)); ref=np.exp(1j*rng.uniform(-np.pi,np.pi,(n,2))); signal=ref*np.exp(1j*phase)
        h=np.column_stack([np.ones(n),s==1,s==2,z,phase]); theta=rng.normal(size=13)
        y=h@theta+.035*rng.normal(size=n); bad=rng.choice(nc,size=120,replace=False); y[bad]+=11*rng.choice([-1,1],size=120)
        x=rng.normal(size=(ns,p))+spread*rng.normal(size=(ns,1)); x/=np.linalg.norm(x,axis=0)
        w=np.zeros(p); w[[1,4,7,10,13]]=[1.9,-.8,1.45,-1.1,.88]
        j=np.arange(nc,nc+ns); y[j]+=x@w
        study={'panel':{'center_id':center,'stratum':s,'X_panel_raw':raw,'controls':control},'interferometry':{'signal_phasor':signal,'ref_phasor':ref},'y_assay_raw':y,'calibration_indices':np.arange(nc),'stage3_indices':j,'X_cand':x,'lam2':.025,'lam0':.16,'M':1.1}
        start=time.monotonic(); result=solve_coupled_interferometry_l0(study); elapsed=time.monotonic()-start
        leaves,gap=verify(result['certified_selection'],x,result['deconfounded_target'],.025,.16,1.1)
        align=np.mean(abs(result['aligned_panel']-z)); rel=np.linalg.norm(result['calibration_weights']-theta)/np.linalg.norm(theta)
        print('full',p,spread,leaves,gap,elapsed,align,rel,flush=True)
        assert align <= .11 and rel <= .14
