import importlib.util
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse, linalg, ndimage
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

path = '/holdout_task/robin_continuation_tomography.py'
spec = importlib.util.spec_from_file_location('robin', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def fixture(kind):
    rng = np.random.default_rng(217 + kind)
    shape = (28, 28, 28) if kind < 2 else (4, 3, 5)
    n = np.prod(shape)
    grids = np.indices(shape)
    v = ndimage.gaussian_filter(rng.normal(size=shape), 1.7) if kind == 0 else rng.random(shape)
    base = np.exp(np.log(3500.) * (v-v.min()) / np.ptp(v)) * 0.02
    if kind == 0:
        m1 = base*rng.uniform(0.01, 0.25, shape)
        m2 = base*rng.uniform(0.01, 0.12, shape)
    else:
        m1 = np.exp(rng.uniform(-5., 3., shape))
        m2 = np.where(grids[1] < shape[1]//2, 25., 0.02)*rng.uniform(0.7, 1.3, shape)
    theta = np.array([1.85, 0.18, 0.98, 0.42, 1.43, 0.72, 1.67, 1.19, 0.15])
    src = np.stack([np.ones(shape), np.sin(4*grids[0]/shape[0])+0.1, rng.normal(size=shape)])
    if kind == 2:
        src[2] *= 0.
    ijk = np.array(np.unravel_index(np.arange(n),shape)).T
    return dict(Lx=1.7,Ly=0.85,Lz=2.3,kappa_base=base,kappa_mod1=m1,kappa_mod2=m2,sigma_base=rng.uniform(0.01,0.5,shape),sigma_mod=rng.uniform(0.01,0.12,shape),gamma_robin=[0.73,0.006,23.][kind],thetas=theta,source_modes=src,sensor_ijk=ijk,nuisance_ridge=1e-3,source_ridge=0.01)

def reference_operator(s, t):
    k = s['kappa_base'] + t*s['kappa_mod1'] + t*t*s['kappa_mod2']
    sig = s['sigma_base'] + t*s['sigma_mod']
    shape = k.shape
    ids = np.arange(k.size).reshape(shape)
    h = np.array([s['Lx'],s['Ly'],s['Lz']])/shape
    vol = np.prod(h)
    gamma = s['gamma_robin']
    wall_reaction = vol*sig.copy()
    matrices = []
    for axis in range(3):
        lo = np.take(ids,range(shape[axis]-1),axis=axis).ravel()
        hi = np.take(ids,range(1,shape[axis]),axis=axis).ravel()
        face_area = np.prod(np.delete(h,axis))
        flat = k.ravel()
        g = face_area / (0.5*h[axis]*(1/flat[lo]+1/flat[hi]))
        ind = np.arange(len(lo))
        grad = sparse.csr_matrix((np.r_[g**0.5,-g**0.5],(np.r_[ind,ind],np.r_[lo,hi])),shape=(len(lo),k.size))
        matrices.append(grad.T@grad)
        for face in [0,-1]:
            slc = [slice(None)]*3
            slc[axis] = face
            idx = tuple(slc)
            wall_reaction[idx] += face_area/(1/gamma+0.5*h[axis]/k[idx])
    return sum(matrices, sparse.diags(wall_reaction.ravel())).tocsc(),vol

elapsed_total = 0.
for kind in range(3):
    s=fixture(kind)
    rng=np.random.default_rng(700+kind)
    mats, ref = [], []
    with threadpool_limits(limits=1,user_api='blas'):
        for t in s['thetas']:
            a, vol = reference_operator(s,t)
            b = vol*s['source_modes'].reshape(3,-1).T
            mats.append(a)
            ref.append(splu(a,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0).solve(b))
        ref = np.asarray(ref)
        design = ref.reshape(-1,3)
        n = design.shape[0]
        z = np.column_stack([np.ones(n),design[:,0]+1e-3*rng.normal(size=n),rng.normal(size=(n,2))])
        y = design @ rng.normal(size=(3,5)) + z @ rng.normal(size=(4,5)) + 1e-4*rng.normal(size=(n,5))
    s.update(Z_nuisance=z,Y_obs=y)
    # Time the complete implementation with full-volume sensors.
    start=perf_counter()
    out=mod.solve_robin_continuation_tomography(s)
    elapsed=perf_counter()-start
    if kind < 2: elapsed_total+=elapsed
    expected_shapes={'terminal_fields':s['source_modes'].shape,'continuation_sensor_predictions':design.shape,'profiled_amplitudes':(3,5),'joint_source_weights':(3,5)}
    assert out.keys()==expected_shapes.keys()
    for key, shape in expected_shapes.items():
        assert out[key].shape == shape and np.isfinite(out[key]).all() and out[key].dtype.kind=='f',key
    states=out['continuation_sensor_predictions'].reshape(ref.shape)
    assert np.array_equal(out['terminal_fields'].reshape(3,-1).T,states[-1])
    with threadpool_limits(limits=1,user_api='blas'):
        den=np.linalg.norm(b,axis=0)
        valid=den>0
        residuals=np.array([np.linalg.norm((a@u-b)[:,valid],axis=0)/den[valid] for a,u in zip(mats,states)])
        err=np.linalg.norm(states-ref)/np.linalg.norm(ref)
        reg_z=np.sqrt(s['nuisance_ridge'])*np.eye(z.shape[1])
        errors=[]
        for m in np.flatnonzero(valid):
            aug=np.block([[z,design[:,m:m+1]],[reg_z,np.zeros((z.shape[1],1))]])
            ref_m=linalg.lstsq(aug,np.vstack([y,np.zeros((z.shape[1],y.shape[1]))]))[0][-1]
            errors.append(np.linalg.norm(out['profiled_amplitudes'][m]-ref_m)/np.linalg.norm(ref_m))
        aug=np.block([[z,design],[reg_z,np.zeros((z.shape[1],3))],[np.zeros((3,z.shape[1])),np.sqrt(s['source_ridge'])*np.eye(3)]])
        joint=linalg.lstsq(aug,np.vstack([y,np.zeros((z.shape[1]+3,y.shape[1]))]))[0][-3:]
        je=np.linalg.norm(out['joint_source_weights']-joint)/np.linalg.norm(joint)
    print(f'case={kind} time_s={elapsed:.3f} residual_max={residuals.max():.3g} residual_argmax={np.unravel_index(residuals.argmax(),residuals.shape)} field_rel={err:.3g} profile_max={max(errors):.3g} joint_rel={je:.3g}',flush=True)
    assert residuals.max() <= 1e-7 and err < 1e-6 and max(errors)<=1e-5 and je<=0.03
print(f'two_large_studies_seconds={elapsed_total:.3f}',flush=True)
assert elapsed_total<=25.
validator = Path('/holdout_task/validate_public_schema.py')
fixture_path = Path('/holdout_task/public_eq01_studies.pkl')
print('public_validator_present',validator.is_file(),'public_fixture_present',fixture_path.is_file())
