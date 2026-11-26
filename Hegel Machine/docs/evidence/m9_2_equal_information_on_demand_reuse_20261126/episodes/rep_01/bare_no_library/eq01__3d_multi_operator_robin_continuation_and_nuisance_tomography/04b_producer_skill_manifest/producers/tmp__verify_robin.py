import importlib.util, pickle, time
from pathlib import Path
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

path = Path('/holdout_task/robin_continuation_tomography.py')
spec = importlib.util.spec_from_file_location('robin', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Independent incidence-matrix assembly, including summed corner Robin fluxes.
def reference_matrix(d,t):
    k=d['kappa_base']+t*d['kappa_mod1']+t**2*d['kappa_mod2']
    shape=k.shape
    h=np.array([d['Lx'],d['Ly'],d['Lz']])/shape
    v=h.prod(); g=d['gamma_robin']; idx=np.arange(k.size).reshape(shape)
    sink=v*(d['sigma_base']+t*d['sigma_mod'])
    cs=[]; ds=[]; ws=[]
    for a in range(3):
        lo=[slice(None)]*3; hi=lo.copy(); lo[a]=slice(None,-1); hi[a]=slice(1,None)
        lo,hi=tuple(lo),tuple(hi)
        cs.append(idx[lo].ravel()); ds.append(idx[hi].ravel())
        ws.append((2*v/h[a]**2/(1/k[lo]+1/k[hi])).ravel())
        for end in (0,-1):
            sl=[slice(None)]*3; sl[a]=end; sl=tuple(sl)
            sink[sl] += (v/h[a])/(1/g+0.5*h[a]/k[sl])
    c=np.concatenate(cs); e=np.concatenate(ds); w=np.concatenate(ws)
    rows=np.repeat(np.arange(len(c)),2)
    cols=np.stack((c,e),axis=1).ravel()
    D=sparse.csr_matrix((np.tile([1.,-1.],len(c)),(rows,cols)),shape=(len(c),k.size))
    return (D.T @ D.multiply(w[:,None]) + sparse.diags(sink.ravel())).tocsc(),v

def make(kind, shape):
    r=np.random.default_rng(941+kind)
    i,j,k=np.indices(shape); X=i/shape[0]; Y=j/shape[1]; Z=k/shape[2]
    def field(lo,hi): return np.exp(r.uniform(np.log(lo),np.log(hi),shape))
    if kind==0:
        base=np.where((i//4+j//5+k//6)%2,12.,12./3500)*field(.8,1.2)
        m1=field(.001,1.); m2=field(.0001,.1)
        lengths=(1.,1.3,.8); gamma=4.
    elif kind==1:
        base=field(1/3500,1.)
        m1=field(.00004,.04); m2=np.where(i<shape[0]//2,2.,.00005)
        lengths=(.42,1.55,.87); gamma=.045
    else:
        base=np.where((j%4==0)&(k%4==0),35.,.01)
        m1=field(.00003,.0003); m2=field(.00001,.0001)
        lengths=(.8,1.4,1.); gamma=.2
    theta=np.array([.15,1.85,.33,1.57,1.13,.66,.91,1.38,.47]); nobs=57
    src=np.array([np.ones(shape), np.cos(2*np.pi*X)*np.sin(np.pi*Y)+.12*r.normal(size=shape), (Z-.4)*np.sin(np.pi*X)+.08*r.normal(size=shape)])
    d=dict(zip(('Lx','Ly','Lz'),lengths)); d.update(kappa_base=base,kappa_mod1=m1,kappa_mod2=m2,sigma_base=field(.008,.08),sigma_mod=field(.001,.012),gamma_robin=gamma,thetas=theta,source_modes=src,sensor_ijk=np.column_stack([r.integers(0,n,nobs) for n in shape]),nuisance_ridge=2e-5,source_ridge=3e-6)
    s,v=truth(d)
    nuisance=r.normal(size=(len(theta)*nobs,12)); nuisance[:,0]=s[:,0]+.0006*nuisance[:,0]; nuisance[:,1]=nuisance[:,2]*2
    y=s@r.normal(size=(3,7))+nuisance@r.normal(size=(12,7))+.007*r.normal(size=(len(s),7))
    d.update(Z_nuisance=nuisance,Y_obs=y)
    return d,(s,v)

def truth(d):
    s=[]; ijk=tuple(d['sensor_ijk'].T); src=d['source_modes']; shape=src.shape[1:]
    for t in d['thetas']:
        a,v=reference_matrix(d,t)
        u=splu(a).solve(v*src.reshape(len(src),-1).T).T.reshape(src.shape)
        s.append(u[(slice(None),)+ijk].T)
    return np.vstack(s),u

def relative(x,y): return np.linalg.norm(x-y)/np.linalg.norm(y)

def check(name,d,expected):
    s,f=expected
    stats=[]; fallback=[]; pcg=mod._pcg; direct=mod.sparse.linalg.spsolve
    def logged(a,b,x,mg,maxiter):
        u,ok=pcg(a,b,x,mg,maxiter)
        if ok: stats.append(relative(a@u,b))
        return u,ok
    def logged_direct(a,b):
        u=direct(a,b); fallback.append(1); stats.append(relative(a@u,b)); return u
    mod._pcg=logged; mod.sparse.linalg.spsolve=logged_direct
    begin=time.perf_counter()
    try: got=mod.solve_robin_continuation_tomography(d)
    finally: mod._pcg=pcg; mod.sparse.linalg.spsolve=direct
    elapsed=time.perf_counter()-begin
    z=d['Z_nuisance']; y=d['Y_obs']; rows,q=z.shape; m=s.shape[1]; lz=d['nuisance_ridge']; ls=d['source_ridge']
    with threadpool_limits(limits=1, user_api='blas'):
        zreg=np.vstack((z,np.sqrt(lz)*np.eye(q)))
        augmented=np.pad(np.column_stack((s,y)),((0,q),(0,0)))
        Q,_=linalg.qr(zreg,mode='economic'); projected=augmented-Q@(Q.T@augmented)
        sp,yp=projected[:,:m],projected[:,m:]
        amps=(sp.T@yp)/np.sum(sp*sp,axis=0)[:,None]
        design=np.block([[z,s],[np.sqrt(lz)*np.eye(q),np.zeros((q,m))],[np.zeros((m,q)),np.sqrt(ls)*np.eye(m)]])
        w=linalg.lstsq(design,np.pad(y,((0,q+m),(0,0))))[0][-m:]
    ae=relative(got['profiled_amplitudes'],amps); we=relative(got['joint_source_weights'],w)
    se=relative(got['continuation_sensor_predictions'],s); fe=relative(got['terminal_fields'],f)
    print(f'{name}: {elapsed:.3f}s, solves={len(stats)}, direct={len(fallback)}, max residual={max(stats):.3g}, stack={se:.3g}, terminal={fe:.3g}, amplitudes={ae:.3g}, joint={we:.3g}',flush=True)
    assert len(stats)==len(d['thetas'])*len(d['source_modes']) and max(stats)<=1e-7 and ae<=1e-5 and we<=.03 and se<1e-6 and fe<1e-6
    return elapsed

if __name__=='__main__':
    fixture=path.with_name('public_eq01_studies.pkl')
    if fixture.exists():
        with fixture.open('rb') as stream: root=pickle.load(stream)
        def studies(node):
            if isinstance(node,dict) and 'kappa_base' in node: yield node
            elif isinstance(node,(dict,list,tuple)):
                for item in (node.values() if isinstance(node,dict) else node): yield from studies(item)
        for i,d in enumerate(studies(root)):
            with threadpool_limits(limits=1,user_api='blas'): expected=truth(d)
            check(f'public-{i}',d,expected)
    else: print('Public fixture unavailable; using independent synthetic references.',flush=True)
    times=[]
    for kind,shape in [(0,(7,6,5)),(0,(28,28,28)),(1,(28,28,28)),(2,(28,28,28))]:
        with threadpool_limits(limits=1,user_api='blas'): d,expected=make(kind,shape)
        elapsed=check(f'kind {kind} {shape}',d,expected)
        if shape[0]==28: times.append(elapsed)
    print(f'slowest full-grid pair: {sum(sorted(times)[-2:]):.3f}s',flush=True)
