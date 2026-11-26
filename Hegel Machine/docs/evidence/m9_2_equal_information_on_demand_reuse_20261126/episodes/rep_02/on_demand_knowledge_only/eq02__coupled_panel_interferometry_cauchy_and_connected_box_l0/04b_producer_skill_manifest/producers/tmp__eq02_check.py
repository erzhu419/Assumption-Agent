import importlib.util, itertools, os, pickle, time
import numpy as np
from scipy.optimize import lsq_linear
p='/holdout_task/coupled_interferometry_certified_l0.py'
sp=importlib.util.spec_from_file_location('eq02',p); mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)

def replay(X,y,l0,l2,M,cert):
    n,p=X.shape; G=X.T@X; w=np.asarray(cert['weights']); err=X@w-y
    obj=.5*(err@err+l2*(w@w))+l0*np.count_nonzero(abs(w)>1e-8)
    assert max(abs(w))<=M and abs(obj-cert['objective'])<1e-9
    leaves=cert['subcube_leaves']; assert 1<=len(leaves)<=2048
    bits=1<<np.arange(p); covers=np.zeros(1<<p,dtype=np.uint8); seen_diag={}; replay_slack=0.
    for leaf in leaves:
        for key in ('fixed_zero','fixed_active'):
            assert len(leaf[key])==p and all(type(v) is bool for v in leaf[key])
        z,a=(np.array(leaf[key]) for key in ('fixed_zero','fixed_active')); assert not (z&a).any()
        free=~(a|z); free_bits=bits[free]; mask=np.array([sum(bits[a])]);
        for b in free_bits: mask=np.r_[mask,mask+b]
        assert not covers[mask].any(); covers[mask]=1
        dl=tuple(leaf['minorant_diagonal']); d=np.array(dl); c=np.array(leaf['minorant_center']); assert c.shape==d.shape==(p,)
        if dl not in seen_diag:
            assert np.min(d)>=0 and np.linalg.eigvalsh(G-np.diag(d))[0]>=-1e-9
            seen_diag[dl]=True
        r=y-X@c; g=X.T@r; t=np.abs(g+d*c); q=d+l2
        b=np.minimum(M,np.divide(t,q,out=np.full(p,M),where=q>0)); on=l0-b*(t-q*b/2)
        replay=.5*(r@r+(d*c)@c)+g@c+on[a].sum()+np.minimum(on[free],0).sum()
        assert leaf['lb']<=replay+1e-8,(leaf['lb'],replay)
        replay_slack=max(replay_slack,replay-leaf['lb'])
    assert covers.all()
    L=min(v['lb'] for v in leaves); assert L==cert['certified_lower_bound'] and 0<=obj-L<=1e-5
    return len(leaves),obj-L,replay_slack

def make(seed,p,corr,bound):
    rng=np.random.default_rng(seed)
    n=850; ncal=630; ns=n-ncal
    group=np.tile(np.arange(5),170); rng.shuffle(group)
    s=np.array([rng.choice(3,p=np.roll([.65,.25,.10],int(g)%3)) for g in group]);
    z=rng.normal(size=(n,8))+.25*s[:,None]; phase=rng.uniform(-2.9,2.9,size=(n,2)); ref=np.exp(1j*rng.uniform(-np.pi,np.pi,size=(n,2)))
    offset=np.array([2,-1,3,-2,.4])[group,None]; gain=np.array([.8,1.1,1.5,2.1,.7])[group,None]
    raw=z*gain+offset; ctrl=rng.normal(size=(n,18))*gain+offset
    raw[rng.random(raw.shape)<.025]=np.nan; ctrl[rng.random(ctrl.shape)<.05]=np.nan
    theta=rng.normal(size=13); H=np.column_stack([np.ones(n),s==1,s==2,z,phase]); y=H@theta+.055*rng.normal(size=n)
    bad=rng.choice(ncal,int(.12*ncal),replace=False); y[bad]+=np.clip(2*rng.standard_cauchy(len(bad)),-11,11)
    X=(np.sqrt(1-corr)*rng.normal(size=(ns,p))+np.sqrt(corr)*rng.normal(size=(ns,1)))/np.sqrt(ns)
    w=np.zeros(p); w[[1,3,6,8,11]]=[.6,-.7,1.2,-1.1,.87] if bound else [.9,-1.5,1.7,-1.5,1.15]; y[ncal:]+=X@w
    return dict(panel=dict(center_id=group,stratum=s,X_panel_raw=raw,controls=ctrl),interferometry=dict(signal_phasor=ref*np.exp(1j*phase),ref_phasor=ref),y_assay_raw=y,calibration_indices=np.arange(ncal),stage3_indices=np.arange(ncal,n),X_cand=X,lam0=.09,lam2=.08,M=.88 if bound else 2.0),z,theta

if __name__=='__main__':
    public='/holdout_task/public_eq02_studies.pkl'
    if os.path.isfile(public):
        studies=pickle.load(open(public,'rb')); print('PUBLIC FIXTURE STRUCTURE',type(studies),list(studies)[:4] if isinstance(studies,dict) else len(studies))
    else: print('PUBLIC FIXTURES UNAVAILABLE')
    for seed in range(12):
        p=18+seed%2; corr=[.02,.4,.78,.91][seed//3]; st,z,theta=make(seed,p,corr,bool(seed%3)); t=time.perf_counter()
        try: out=mod.solve_coupled_interferometry_l0(st)
        except ArithmeticError as e:
            print('FAIL', seed,p,corr,round(time.perf_counter()-t,3),str(e)); continue
        elapsed=time.perf_counter()-t; mae=np.abs(out['aligned_panel']-z).mean(); rel=np.linalg.norm(out['calibration_weights']-theta)/np.linalg.norm(theta)
        cert=out['certified_selection']; print('SLICE',seed,p,corr,'seconds',round(elapsed,4),'mae',round(mae,4),'rel',round(rel,4),'cert',replay(st['X_cand'],out['deconfounded_target'],st['lam0'],st['lam2'],st['M'],cert),flush=True)
    for seed in range(4):
        rng=np.random.default_rng(seed+20); X=rng.normal(size=(23,9)); X[:,1]+=.9*X[:,0]; y=rng.normal(size=23); M=.18+seed*.08; l0=.30; l2=.03
        cert=mod._box_l0(X,y,l0,l2,M); replay(X,y,l0,l2,M,cert); ref=np.dot(y,y)/2
        for k in range(1,10):
            for inds in itertools.combinations(range(9),k):
                A=np.vstack((X[:,inds],np.sqrt(l2)*np.eye(k))); b=np.r_[y,np.zeros(k)]; z=lsq_linear(A,b,bounds=(-M,M),tol=1e-13).x
                r=A@z-b; ref=min(ref,.5*(r@r)+k*l0)
        assert abs(ref-cert['objective'])<1e-7,(ref,cert['objective']); print('ORACLE',seed,'error',abs(ref-cert['objective']),flush=True)
