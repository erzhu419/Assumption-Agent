import importlib.util
import itertools
import os
import time
import numpy as np

path = '/holdout_task/coupled_interferometry_certified_l0.py'
spec = importlib.util.spec_from_file_location('eq02', path)
eq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eq)

def check(x,y,l2,l0,m,out):
    p=x.shape[1]; w=np.asarray(out['weights']); r=x@w-y
    value=.5*(r@r+l2*(w@w))+l0*np.count_nonzero(abs(w)>1e-8)
    assert w.shape==(p,) and np.max(abs(w))<=m+1e-12 and abs(value-out['objective'])<1e-8
    leaves=out['subcube_leaves']; assert 1<=len(leaves)<=2048
    universe=np.arange(1<<p,dtype=np.int32); covered=np.zeros(len(universe),dtype=np.int16)
    g=x.T@x; b=x.T@y; base=.5*(y@y); smallest=np.inf; slack=np.inf
    for leaf in leaves:
        z,a=[leaf[k] for k in ('fixed_zero','fixed_active')]
        assert all(len(v)==p and all(type(q) is bool for q in v) for v in (z,a))
        z,a=np.asarray(z),np.asarray(a); assert not (z&a).any()
        ab=sum(1<<int(j) for j in np.flatnonzero(a)); zb=sum(1<<int(j) for j in np.flatnonzero(z))
        covered[(universe&(ab|zb))==ab]+=1
        d,c=[np.asarray(leaf[k]) for k in ('minorant_diagonal','minorant_center')]
        assert d.shape==c.shape==(p,) and np.all(d>=0)
        q=g-np.diag(d); slack=min(slack,np.linalg.eigvalsh(q)[0]); assert slack>=-1e-9
        coef=q@c-b; curv=d+l2
        t=np.array([np.clip(-u/v,-m,m) if v>0 else -np.sign(u)*m for u,v in zip(coef,curv)])
        f=l0+coef*t+.5*curv*t*t
        recomputed=base-.5*c@(q@c)+sum(f[a])+sum(np.minimum(f[~(a|z)],0))
        assert leaf['lb']<=recomputed+1e-8
        smallest=min(smallest,leaf['lb'])
    assert np.all(covered==1)
    gap=value-out['certified_lower_bound']; assert 0<=gap<=1e-5 and abs(out['certified_lower_bound']-smallest)<1e-8
    return len(leaves),gap,slack

def design(rng,n,p,rho):
    x=rng.normal(size=(n,p))
    if rho:
        x=np.sqrt(1-rho)*x+np.sqrt(rho)*rng.normal(size=(n,1))*rng.choice([-1,1],size=p)
    return x/np.linalg.norm(x,axis=0)*rng.uniform(1.7,3.0,p)

def panel_trial(seed,p,rho):
    rng=np.random.default_rng(seed); n=980; ns=210
    probs=np.geomspace(1,7,15); probs/=sum(probs)
    cells=np.r_[np.arange(15),rng.choice(15,n-15,p=probs)]; rng.shuffle(cells)
    c,s=cells//3,cells%3
    c_names=np.array([17,29,8,41,94])[c]
    gain=np.array([.52,.85,1.1,1.6,2.0])[c,None]; offset=np.array([-1.4,.0,1.3,1.8,-.8])[c,None]
    x=rng.normal(size=(n,8))+s[:,None]*rng.normal(0,.27,(1,8)); raw=x*gain+offset
    controls=rng.normal(size=(n,18))*gain+offset
    raw[rng.random(raw.shape)<.025]=np.nan; controls[rng.random(controls.shape)<.05]=np.nan
    controls[c==0,0]=np.nan; controls[np.flatnonzero(c==0)[0],0]=1000
    freq=np.bincount(cells); wt=1/freq[cells]; expected=raw.copy()
    for k in range(5):
        cc=controls[c==k]; cols=[v[np.isfinite(v)] for v in cc.T if np.isfinite(v).sum()>=2]
        expected[c==k]=(raw[c==k]-np.median([v.mean() for v in cols]))/np.median([v.std() for v in cols])
    for j in range(8):
        ok=np.isfinite(expected[:,j]); expected[~ok,j]=np.average(expected[ok,j],weights=wt[ok])
    phase=rng.uniform(-1.5,1.5,(n,2)); carrier=rng.uniform(-np.pi,np.pi,(n,2)); ref=1.35*np.exp(1j*carrier); sig=.88*np.exp(1j*(carrier+phase))
    beta=rng.uniform(-.9,.9,13); h=np.column_stack([np.ones(n),s==1,s==2,x,phase]); target=rng.choice(n,ns,replace=False); cal=np.setdiff1d(np.arange(n),target)
    y=h@beta+rng.normal(0,.11,n); glitches=rng.choice(cal,round(.12*len(cal)),replace=False); y[glitches]+=np.clip(4*rng.standard_cauchy(len(glitches)),-11,11)
    xx=design(rng,ns,p,rho); signal=np.zeros(p); idx=rng.choice(p,5,replace=False); signal[idx]=rng.uniform(.60,1.55,len(idx))*rng.choice([-1,1],len(idx)); y[target]+=xx@signal
    study=dict(panel=dict(center_id=c_names,stratum=s,X_panel_raw=raw,controls=controls),interferometry=dict(signal_phasor=sig,ref_phasor=ref), y_assay_raw=y,calibration_indices=cal,stage3_indices=target,X_cand=xx,lam2=.08,lam0=.28,M=1.1)
    start=time.perf_counter(); out=eq.solve_coupled_interferometry_l0(study); elapsed=time.perf_counter()-start
    aligned=out['aligned_panel']; theta=out['calibration_weights']; dy=out['deconfounded_target']
    np.testing.assert_allclose(aligned,expected,atol=2e-14)
    np.testing.assert_allclose(dy,y[target]-np.column_stack([np.ones(n),s==1,s==2,aligned,phase])[target]@theta,atol=2e-14)
    mae=np.mean(abs(aligned-x)); rel=np.linalg.norm(theta-beta)/np.linalg.norm(beta)
    summary=check(xx,dy,.08,.28,1.1,out['certified_selection']); print('panel',seed,p,rho,'MAE',round(mae,6),'relL2',round(rel,6),'leaves/gap/psd',summary,'seconds',round(elapsed,4),flush=True)
    assert mae<=.11 and rel<=.14
    return elapsed

def small_oracle():
    rng=np.random.default_rng(42); x=design(rng,24,6,.71); y=rng.normal(size=24)*1.6; l2=.1; l0=.21; m=.35
    output=eq._Selector(x,y,l2,l0,m).solve(); ref=np.inf
    g=x.T@x+l2*np.eye(6); b=x.T@y
    for face in itertools.product((0,1,2,3),repeat=6):
        f=np.array(face); w=np.zeros(6); w[f==2]=-m; w[f==3]=m; free=f==1
        if free.any():
            w[free]=np.linalg.solve(g[np.ix_(free,free)], b[free]-g[np.ix_(free,~free)]@w[~free])
        if np.max(abs(w))>m: continue
        r=x@w-y; ref=min(ref,.5*(r@r+l2*w@w)+l0*np.count_nonzero(abs(w)>1e-8))
    assert abs(ref-output['objective'])<=1e-5
    print('face oracle error',abs(ref-output['objective']),'leaves/gap/psd',check(x,y,l2,l0,m,output),flush=True)

if __name__=='__main__':
    if os.path.isfile('/holdout_task/public_eq02_studies.pkl'):
        print('Public fixtures now present',flush=True)
    small_oracle()
    elapsed=[panel_trial(k,p,r) for k,p,r in [(91,18,.34),(92,19,.72),(93,19,.86),(96,18,.92)]]
    assert sum(sorted(elapsed)[-2:])<=18
