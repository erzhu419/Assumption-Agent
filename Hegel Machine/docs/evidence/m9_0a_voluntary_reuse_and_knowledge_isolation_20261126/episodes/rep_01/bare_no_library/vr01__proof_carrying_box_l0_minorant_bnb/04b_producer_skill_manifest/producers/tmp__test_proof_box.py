import importlib.util, itertools, pickle, time
import numpy as np
s = importlib.util.spec_from_file_location('proof_box_l0','/holdout_task/proof_box_l0.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def check(p, gt=None):
    st=time.monotonic(); r=m.solve_proof_carrying_box_l0(p); elapsed=time.monotonic()-st
    X,y,l2,l0,M=(p[k] for k in ('X','y','lam2','lam0','M')); G=X.T@X; n,k=X.shape
    w=r['weights']; f=.5*np.linalg.norm(y-X@w)**2+.5*l2*(w@w)+l0*np.count_nonzero(abs(w)>1e-8)
    L=r['certified_lower_bound']; leaves=r['subcube_leaves']; slack=float('inf'); psd=float('inf'); covered=0; masks=[]; seen=set()
    assert w.shape==(k,) and np.max(abs(w))<=M and np.all((w==0)|(abs(w)>1e-8))
    assert abs(f-r['objective'])<1e-9 and -1e-10<=f-L<=1e-5 and 1<=len(leaves)<=2048
    if gt is not None: assert abs(f-gt)<1e-5, (f,gt)
    for leaf in leaves:
        z,a=(np.asarray(leaf[key]) for key in ('fixed_zero','fixed_active')); u=~(z|a)
        assert z.shape==a.shape==(k,) and z.dtype==a.dtype==np.bool_ and not (z&a).any()
        covered+=1<<int(u.sum())
        masks.append((sum(1<<int(j) for j in np.flatnonzero(z)),sum(1<<int(j) for j in np.flatnonzero(a))))
        d,c=(np.asarray(leaf[key]) for key in ('minorant_diagonal','minorant_center'))
        assert d.shape==c.shape==(k,) and d.min()>=0 and np.isfinite(c).all()
        dk=tuple(d)
        if dk not in seen:
            seen.add(dk); psd=min(psd,np.linalg.eigvalsh(G-np.diag(d))[0]); assert psd>=-1e-8
        v=y-X@c; g=abs(X.T@v+d*c); b=np.minimum(g/(l2+d),M); s=l0-g*b+.5*(l2+d)*b**2
        bound=.5*np.linalg.norm(v)**2+v@(X@c)+.5*(d*c)@c+s[a].sum()+np.minimum(0,s[u]).sum()
        slack=min(slack,bound-leaf['lb']); assert leaf['lb']<=bound+1e-7 and L<=leaf['lb']+1e-6
    assert covered==1<<k
    assert all((z&aa)|(a&zz) for (z,a),(zz,aa) in itertools.combinations(masks,2))
    return len(leaves), f-L, slack, psd, elapsed, f
if __name__=='__main__':
    for a in pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb')):
        print('public:',check(a['problem'],a['gt']['exact_objective']),flush=True)
