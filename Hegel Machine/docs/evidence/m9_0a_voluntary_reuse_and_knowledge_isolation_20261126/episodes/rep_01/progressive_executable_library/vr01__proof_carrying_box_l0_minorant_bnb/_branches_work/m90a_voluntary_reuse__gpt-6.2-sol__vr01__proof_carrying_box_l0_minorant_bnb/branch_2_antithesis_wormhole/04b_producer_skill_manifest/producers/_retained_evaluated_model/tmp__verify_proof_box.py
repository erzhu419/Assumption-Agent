import importlib.util, pickle, time
import numpy as np
from itertools import combinations
path='/holdout_task/proof_box_l0.py'
spec=importlib.util.spec_from_file_location('proof_box_l0',path)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)

def verify(p,r,exact=None):
    X,y,l0,l2,M=[p[k] for k in ('X','y','lam0','lam2','M')]
    G=X.T@X; w=r['weights']; L=r['certified_lower_bound']; U=r['objective']; leaves=r['subcube_leaves']; n=X.shape[1]
    assert w.shape==(n,) and np.all(np.abs(w)<=M)
    assert abs(U-(.5*np.sum((X@w-y)**2)+.5*l2*(w@w)+l0*np.count_nonzero(abs(w)>1e-8)))<=1e-9
    assert 0<=U-L<=1e-5 and 1<=len(leaves)<=2048
    if exact is not None: assert abs(U-exact)<=1e-5 and L<=exact+1e-9,(U,L,exact)
    zmas=[]; amas=[]; volume=0; slack=np.inf; eig=np.inf
    for leaf in leaves:
        z,A=np.asarray(leaf['fixed_zero']),np.asarray(leaf['fixed_active'])
        assert z.shape==A.shape==(n,) and z.dtype==A.dtype==bool and not np.any(z&A)
        F=~(z|A); volume+=1<<int(F.sum())
        zmas.append(sum(1<<i for i in np.flatnonzero(z).tolist())); amas.append(sum(1<<i for i in np.flatnonzero(A).tolist()))
        d,c=leaf['minorant_diagonal'],leaf['minorant_center']; rr=y-X@c
        v=np.linalg.eigvalsh(G-np.diag(d))[0]; eig=min(eig,v)
        assert d.shape==c.shape==(n,) and np.all(d>=0) and v>=-1e-10,(v,d)
        g=np.abs(X.T@rr+d*c); b=np.clip(g/(l2+d),0.,M); a=l0-g*b+.5*(l2+d)*b*b
        lb=.5*np.sum(rr**2)+rr@X@c+.5*np.sum(d*c*c)+sum(a[A])+sum(np.minimum(a[F],0.))
        slack=min(slack,lb-leaf['lb'])
        assert L<=leaf['lb']+1e-6 and leaf['lb']<=lb+1e-10,(lb,leaf['lb'])
    assert volume==1<<n and all((zmas[i]&amas[j]) or (zmas[j]&amas[i]) for i,j in combinations(range(len(leaves)),2))
    return len(leaves), U-L, eig, slack

def run(label,p,exact=None):
    t=time.perf_counter()
    try: print(label,verify(p,s.solve_proof_carrying_box_l0(p),exact),'sec',round(time.perf_counter()-t,4),flush=True)
    except Exception as e: print(label,type(e).__name__,str(e),'sec',round(time.perf_counter()-t,4),flush=True)

def make(X,y,l0=.22,l2=.14,M=1.1): return dict(X=X,y=y,lam0=l0,lam2=l2,M=M)
if __name__=='__main__':
    for i,c in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb'))): run(f'public-{i}',c['problem'],c['gt']['exact_objective'])
    X=np.eye(22); y=np.full(22,np.sqrt(2*.22*1.14)); run('22 orthogonal ties',make(X,y),.5*(y@y))
    y=np.linspace(-3,3,22); b=np.minimum(.25,abs(y)/1.14); run('box face',make(X,y,M=.25),.5*(y@y)+np.minimum(0,.22-abs(y)*b+.5*1.14*b*b).sum())
    for k in (9,11):
        X=np.kron(np.eye(k),[[1.,1.]]); X=np.pad(X,((0,0),(0,max(0,19-2*k)))); y=np.ones(k); run(f'{k} duplicate pairs',make(X,y),k*(.5+.22-.5/1.14))
    X=np.zeros((20,22)); X[0,:2]=[2.,.5]; X[1:19,2:20]=np.eye(18); d=s._diagonal(X.T@X); print('rank-deficient unequal factor: residual minimum eigenvalue',np.linalg.eigvalsh(X.T@X-np.diag(d))[0],flush=True)
    for strength in (.35,.65):
        for seed in range(4):
            rng=np.random.default_rng(seed); A=rng.normal(size=(22,22)); np.fill_diagonal(A,0); A+=A.T; G=np.eye(22)+strength*A/np.linalg.norm(A,2); X=np.linalg.cholesky(G).T
            y=np.linalg.solve(X.T,rng.choice([-1,1],22)*rng.uniform(.55,.85,22)); run(f'dense {strength} seed {seed}',make(X,y))
