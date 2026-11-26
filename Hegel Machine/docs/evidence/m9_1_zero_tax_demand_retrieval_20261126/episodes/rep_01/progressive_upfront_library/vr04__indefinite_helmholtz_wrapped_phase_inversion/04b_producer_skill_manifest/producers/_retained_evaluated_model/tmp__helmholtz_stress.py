import importlib.util
import pickle
from copy import deepcopy

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

spec = importlib.util.spec_from_file_location('helmholtz', '/holdout_task/helmholtz_phase_inversion.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_helmholtz_cases.pkl', 'rb') as f:
    cases = pickle.load(f)

def errors(p, a, phi, field):
    out = solver.invert_helmholtz_sources_and_phase(p)
    assert set(out) == {'source_amplitudes', 'global_phase_rad', 'predicted_field_abs'}
    assert out['source_amplitudes'].shape == a.shape and out['source_amplitudes'].dtype.kind == 'f'
    assert out['predicted_field_abs'].shape == field.shape and out['predicted_field_abs'].dtype.kind == 'f'
    assert isinstance(out['global_phase_rad'], float) and -np.pi < out['global_phase_rad'] <= np.pi
    return np.array([np.linalg.norm(out['source_amplitudes']-a)/np.linalg.norm(a),
                     abs(np.angle(np.exp(1j*(out['global_phase_rad']-phi)))),
                     np.linalg.norm(out['predicted_field_abs']-abs(field))/np.linalg.norm(field)])

def reference(p, a):
    nx, ny = p['n_sq'].shape
    hx, hy = p['Lx']/nx, p['Ly']/ny
    k = p['k0']
    op = lil_matrix((nx*ny, nx*ny), dtype=complex)
    for i,j in np.ndindex(nx,ny):
        t = i*ny+j
        op[t,t] = -hx*hy*k*k*p['n_sq'][i,j]
        for ii,jj,length,distance in [(i-1,j,hy,hx),(i+1,j,hy,hx),(i,j-1,hx,hy),(i,j+1,hx,hy)]:
            if 0 <= ii < nx and 0 <= jj < ny:
                op[t,t] += length/distance
                op[t,ii*ny+jj] -= length/distance
            else:
                op[t,t] -= 1j*k*p['beta_imp']*length
    q = np.einsum('m,mij->ij',a,p['basis'])
    return spsolve(op.tocsc(), hx*hy*q.ravel()).reshape(nx,ny)

rng = np.random.default_rng(2047)
loo, synthetic = [], []
for case in cases:
    p, gt = case['problem'], case['gt']
    a, phi, field = (gt[k] for k in ('source_amplitudes','global_phase_rad','predicted_field_abs'))
    for dropped in range(len(p['survey']['sensor_ij'])):
        q = dict(p, survey={k: np.delete(v, dropped, axis=0) if k != 'tikhonov_alpha' else v for k,v in p['survey'].items()})
        loo.append(errors(q, a, phi, field))
    for lx, ly, k0, beta in [(0.82,1.20,10.9,0.65),(1.21,0.78,12.7,1.02)]:
        q = deepcopy(p)
        q.update(Lx=lx, Ly=ly, k0=k0, beta_imp=beta)
        u = reference(q,a)
        np.testing.assert_allclose(solver._source_fields(q) @ a, u, rtol=2e-11, atol=1e-12)
        pilot, survey = q['pilot_calibration'],q['survey']
        ref = pilot['known_field_re']+1j*pilot['known_field_im']
        i,j = survey['sensor_ij'].T
        for phi in [-np.pi+1e-5, np.pi-1e-5]:
            zpilot = np.exp(1j*phi)*ref + .0015*(rng.normal(size=ref.size)+1j*rng.normal(size=ref.size))
            pilot.update(amplitude_obs=abs(zpilot),wrapped_phase_obs=np.angle(zpilot))
            for repeat in range(12):
                z = np.exp(1j*phi)*u[i,j] + .0015*(rng.normal(size=i.size)+1j*rng.normal(size=i.size))
                nbad = round(.07*len(z))
                bad = rng.choice(len(z),nbad,replace=False) if repeat%2 else np.argsort(abs(u[i,j]))[-nbad:]
                z[bad] += rng.uniform(.15,.75,nbad)*np.exp(1j*rng.uniform(-np.pi,np.pi,nbad))
                survey.update(amplitude_obs=abs(z),wrapped_phase_obs=np.angle(z))
                synthetic.append(errors(q,a,phi,u))
for label, vals in [('public leave-one-receiver-out',loo),('rectangular / branch-cut / random and high-signal impulses',synthetic)]:
    worst = np.max(vals,axis=0)
    print(label, len(vals), 'worst source / phase / field:',worst)
    assert np.all(worst <= [.03,.02,.03]), (label,worst)
