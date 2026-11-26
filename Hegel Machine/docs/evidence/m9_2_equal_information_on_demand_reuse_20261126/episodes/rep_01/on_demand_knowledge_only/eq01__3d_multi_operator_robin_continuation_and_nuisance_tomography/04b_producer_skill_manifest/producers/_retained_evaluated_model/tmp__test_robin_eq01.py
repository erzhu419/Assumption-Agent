import importlib.util
import pickle
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

path = Path('/holdout_task/robin_continuation_tomography.py')
spec = importlib.util.spec_from_file_location('robin', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def generate(seed, shape, kind):
    rng = np.random.default_rng(seed)
    x,y,z = np.meshgrid(*(np.linspace(-1,1,n) for n in shape), indexing='ij')
    if kind == 0:
        k = 0.03 * 3500**rng.random(shape)
        k1 = 0.2*k*rng.random(shape)
        k2 = k*rng.random(shape)
        gamma, reaction = 0.9, 0.08
    else:
        k = np.where(((x+0.3*z)**2+(y-0.15)**2 < 0.3) | (abs(z)<0.067),105.,0.03)
        k1 = np.where(y>0,0.4,0.001)
        k2 = np.where(x>0,18.,0.0001)
        gamma, reaction = 0.004, 0.00015
    theta = np.array([0.82,0.15,1.7,0.24,1.02,1.85,0.4,1.31,0.59])
    sensors = np.column_stack([rng.integers(n,size=100) for n in shape])
    Z = rng.normal(size=(900,10)); Z[:,5] = Z[:,2]+Z[:,3]
    Z[:,0] = 1.
    Y = rng.normal(size=(900,6)) + Z @ rng.normal(size=(10,6))
    return dict(Lx=1.25,Ly=0.8,Lz=1.7,kappa_base=k,kappa_mod1=k1,kappa_mod2=k2,
                sigma_base=reaction*(1+rng.random(shape)),sigma_mod=reaction*rng.random(shape),gamma_robin=gamma,
                thetas=theta,source_modes=np.array([1+0.3*x,np.exp(-7*((x+0.2)**2+y*y+z*z)),(y+0.4*z)*np.cos(3*x)]),
                sensor_ijk=sensors,Z_nuisance=Z,Y_obs=Y,nuisance_ridge=1e-4,source_ridge=0.001)

def oracle_operator(s, theta):
    k=s['kappa_base']+theta*s['kappa_mod1']+theta**2*s['kappa_mod2']
    sig=s['sigma_base']+theta*s['sigma_mod']; g=s['gamma_robin']
    h=np.array([s['Lx'],s['Ly'],s['Lz']])/k.shape
    v=h.prod(); ids=np.arange(k.size).reshape(k.shape)
    diag=v*sig.copy(); weights=[]; left=[]; right=[]
    for axis in range(3):
        K=np.moveaxis(k,axis,0); I=np.moveaxis(ids,axis,0); D=np.moveaxis(diag,axis,0)
        face=(v/h[axis])/(0.5*h[axis]*(1/K[:-1]+1/K[1:]))
        weights.append(face.ravel()); left.append(I[:-1].ravel()); right.append(I[1:].ravel())
        D[0] += (v/h[axis])/(1/g+0.5*h[axis]/K[0]); D[-1] += (v/h[axis])/(1/g+0.5*h[axis]/K[-1])
    f=np.concatenate(weights); i=np.concatenate(left); j=np.concatenate(right); r=np.arange(len(f))
    incidence=sparse.csr_matrix((np.tile([1.,-1.],(len(f),1)).ravel(), (np.repeat(r,2), np.column_stack((i,j)).ravel())),shape=(len(f),k.size))
    return (incidence.T @ incidence.multiply(f[:,None]) + sparse.diags(diag.ravel())).tocsc(), v

def error(actual, expected):
    return np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-100)

def oracle_profile(s, S):
    Z,Y=s['Z_nuisance'],s['Y_obs']; q=Z.shape[1]; p=S.shape[1]
    rz=np.sqrt(s['nuisance_ridge']); rs=np.sqrt(s['source_ridge'])
    az=np.vstack((Z,rz*np.eye(q))); ay=np.vstack((Y,np.zeros((q,Y.shape[1]))))
    mode=[]
    for x in S.T:
        design=np.column_stack((az,np.r_[x,np.zeros(q)]))
        mode.append(linalg.lstsq(design,ay,lapack_driver='gelsy')[0][-1])
    design=np.block([[Z,S],[rz*np.eye(q),np.zeros((q,p))],[np.zeros((p,q)),rs*np.eye(p)]])
    rhs=np.vstack((ay,np.zeros((p,Y.shape[1]))))
    return np.array(mode),linalg.lstsq(design,rhs,lapack_driver='gelsy')[0][-p:]

def test(s, label, exact):
    # Capture internal solves so residuals can be checked for nonterminal states, too.
    orig = m._iterate; xs = []
    def save(A,b,M,x):
        field=orig(A,b,M,x); xs.append(field); return field
    m._iterate = save
    start=perf_counter()
    try: out=m.solve_robin_continuation_tomography(s)
    finally: m._iterate=orig
    sec=perf_counter()-start
    ks=s['thetas']; p=len(s['source_modes']); sz=s['kappa_base'].shape
    xs=np.asarray(xs).reshape(len(ks),p,-1)
    order=np.argsort(ks); sensors=tuple(s['sensor_ijk'].T); pred=[None]*len(ks); ref=[None]*len(ks)
    res=[]
    for fields,k in zip(xs,order):
        A,v=oracle_operator(s,ks[k]); b=(v*s['source_modes']).reshape(p,-1).T
        res.extend(np.linalg.norm(A@fields.T-b,axis=0)/np.maximum(np.linalg.norm(b,axis=0),1e-100))
        pred[k]=np.column_stack([u.reshape(sz)[sensors] for u in fields]);
        if exact:
            sol=splu(A,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,options={'SymmetricMode':True}).solve(b)
            ref[k]=np.column_stack([u.reshape(sz)[sensors] for u in sol.T]);
    assert set(out)=={'terminal_fields','continuation_sensor_predictions','profiled_amplitudes','joint_source_weights'}
    np.testing.assert_array_equal(out['terminal_fields'],xs[np.flatnonzero(order==len(ks)-1)[0]].reshape((p,)+sz))
    S=np.vstack(pred)
    np.testing.assert_array_equal(out['continuation_sensor_predictions'],S)
    truth=np.vstack(ref) if exact else S
    a,w=oracle_profile(s,truth)
    ea,ew=error(out['profiled_amplitudes'],a),error(out['joint_source_weights'],w)
    print(label, f'{sec:.3f}s worst residual={max(res):.3g} sensor L2={error(S,truth):.3g} amplitude L2={ea:.3g} joint L2={ew:.3g}', flush=True)
    assert max(res)<=1e-7 and ea<=1e-5 and ew<=0.03
    return sec

if __name__ == '__main__':
    public=Path('/holdout_task/public_eq01_studies.pkl')
    if public.exists():
        data=pickle.loads(public.read_bytes()); print('Public pickle type:',type(data), list(data)[:8] if isinstance(data,dict) else len(data))
    else:
        print('Public study unavailable; testing synthetic studies with independent discretization and augmented QR oracles.',flush=True)
    test(generate(741,(5,4,6),1),'small anisotropic',True)
    t1=test(generate(975,(28,28,28),0),'3500x cell-scale contrast',True)
    t2=test(generate(741,(28,28,28),1),'3500x channels, weak walls, shifting coefficients',True)
    print(f'Full-size pair runtime: {t1+t2:.3f} seconds',flush=True)
    assert t1+t2 <= 25
