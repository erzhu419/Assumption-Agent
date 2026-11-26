import importlib.util, itertools, os, pickle, time
import numpy as np
from scipy.optimize import minimize
path='/holdout_task/coupled_interferometry_certified_l0.py'
spec=importlib.util.spec_from_file_location('eq02',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def verify(x,y,l2,l0,M,cert):
    w=np.asarray(cert['weights']); p=len(w); leaves=cert['subcube_leaves']; G=x.T@x; b=x.T@y; const=y@y/2
    assert w.shape==(x.shape[1],) and np.max(abs(w))<=M+1e-10
    f=np.linalg.norm(x@w-y)**2/2+l2*(w@w)/2+l0*np.count_nonzero(abs(w)>1e-8)
    assert abs(cert['objective']-f)<1e-8
    ids=np.arange(1<<p,dtype=np.int32); cover=np.zeros(len(ids),dtype=np.uint16); min_lb=np.inf; diags={}
    bits=1<<np.arange(p)
    for leaf in leaves:
        assert all(len(leaf[k])==p and all(type(v) is bool for v in leaf[k]) for k in ('fixed_zero','fixed_active'))
        z=np.asarray(leaf['fixed_zero']); a=np.asarray(leaf['fixed_active']); assert not np.any(z&a)
        mask=int(bits@(z|a)); active=int(bits@a); free=((1<<p)-1)^mask
        # Enumerate only this cube's vertices.
        sub=np.array([0],dtype=np.int32)
        while free:
            bit=free&-free; sub=np.r_[sub,sub|bit]; free^=bit
        cover[sub|active]+=1
        d=np.asarray(leaf['minorant_diagonal']); u=np.asarray(leaf['minorant_center']); key=tuple(d)
        if key not in diags:
            assert np.min(d)>=0 and np.linalg.eigvalsh(G-np.diag(d))[0]>=-1e-8
            diags[key]=G-np.diag(d)
        B=diags[key]; h=B@u-b; q=d+l2
        v=np.clip(-h/q,-M,M); part=l0+h*v+q*v*v/2
        lb=const-u@B@u/2+sum(part[a])+sum(np.minimum(part[~(a|z)],0))
        assert abs(lb-leaf['lb'])<1e-8*(1+abs(lb))
        min_lb=min(min_lb,lb)
    assert np.all(cover==1)
    gap=f-cert['certified_lower_bound']; assert abs(cert['certified_lower_bound']-min(min_lb,cert['objective']))<1e-8
    return f,gap,len(leaves)

def fixture(seed,p,coupling,M):
    rng=np.random.default_rng(seed)
    counts=rng.integers(22,65,(5,3)); c,s=np.repeat(np.arange(15),counts.ravel())//3,np.repeat(np.arange(15),counts.ravel())%3
    N=len(c); T=rng.normal(size=(N,8))+s[:,None]*rng.normal(0,.25,(1,8)); R=rng.normal(size=(N,18)); shift=np.linspace(-1.8,1.8,5)[c]; gain=np.linspace(.55,1.6,5)[c]
    raw=shift[:,None]+gain[:,None]*T; raw[rng.random(raw.shape)<.025]=np.nan
    R=shift[:,None]+gain[:,None]*R; R[rng.random(R.shape)<.05]=np.nan
    phase=rng.uniform(-1.5,1.5,(N,2)); ref=np.exp(1j*rng.uniform(-np.pi,np.pi,(N,2))); signal=ref*np.exp(1j*phase)
    h=np.column_stack((np.ones(N),s==1,s==2,T,phase)); theta=rng.normal(0,.35,13); yy=h@theta+rng.normal(0,.08,N)
    loc=rng.permutation(N); cal,j=loc[110:],loc[:110]; glitch=rng.choice(cal,round(.12*len(cal)),replace=False); yy[glitch]+=np.clip(2.5*rng.standard_cauchy(len(glitch)),-11,11)
    X=rng.normal(size=(len(j),p)); X+=coupling*rng.normal(size=(len(j),1)); X/=np.linalg.norm(X,axis=0)
    sparse=np.zeros(p); sparse[:5]=np.minimum(M,1.4)*np.array([1,-.9,1,-.75,.85]); yy[j]+=X@sparse
    st=dict(panel=dict(center_id=c,stratum=s,X_panel_raw=raw,controls=R),interferometry=dict(signal_phasor=signal,ref_phasor=ref), y_assay_raw=yy,calibration_indices=cal,stage3_indices=j,X_cand=X,lam2=.045,lam0=.085,M=M)
    return st,T,theta

if __name__=='__main__':
    rng=np.random.default_rng(538)
    for k in range(4):
        x=rng.normal(size=(30,8))+k*.65*rng.normal(size=(30,1)); x/=np.linalg.norm(x,axis=0); y=rng.normal(0,.4,30); l2,l0,M=.04,.08,.15+.3*k
        cert=mod._box_l0(x,y,l2,l0,M); f,gap,n=verify(x,y,l2,l0,M,cert); best=y@y/2
        for r in range(1,9):
            for ix in itertools.combinations(range(8),r):
                A=x[:,ix]; Q=A.T@A+l2*np.eye(r); h=A.T@y
                def fun(w): return (Q@w/2-h)@w,Q@w-h
                R=minimize(fun,np.zeros(r),jac=True,bounds=[(-M,M)]*r,method='L-BFGS-B',options={'gtol':1e-11,'ftol':1e-15})
                best=min(best,y@y/2+R.fun+l0*r)
        assert abs(best-f)<1e-7 and -1e-10<=gap<=1e-5
        print('exhaustive',k,'objective error',f-best,'gap',gap,'leaves',n,flush=True)
    for k,(p,c,M) in enumerate([(18,.45,1.4),(19,1.3,.63),(19,3,1.1),(18,.8,.32),(19,5,.35)]):
        st,T,theta=fixture(876+k,p,c,M); t=time.monotonic(); out=mod.solve_coupled_interferometry_l0(st); t=time.monotonic()-t
        f,gap,n=verify(st['X_cand'],out['deconfounded_target'],st['lam2'],st['lam0'],M,out['certified_selection']); print(f'fixture {k}: p={p} coupling={c} seconds={t:.3f} MAE={np.mean(abs(out["aligned_panel"]-T)):.5f} theta_err={np.linalg.norm(out["calibration_weights"]-theta)/np.linalg.norm(theta):.5f} gap={gap:.3g} leaves={n}',flush=True)
    public='/holdout_task/public_eq02_studies.pkl'
    if os.path.isfile(public): print('public fixtures now available',flush=True)
