import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
maxima=np.zeros(4)
def verify(S,y,r,u,b,c):
    global maxima
    w=c['weights']; l=c['lam_lower']; h=c['lam_upper']; nu=c['nu_sum']; g=S.T@(S@w-y)+r*w
    p=.5*(np.linalg.norm(S@w-y)**2+r*(w@w))
    assert np.all(l>=0) and np.all(h>=0) and abs(p-c['primal_objective'])<1e-9
    gap=p-c['dual_lower_bound']; assert gap>=-1e-9
    err=np.array([max(abs(w.sum()-b),float(np.max(-w)),float(np.max(w-u))), np.max(abs(g+nu-l+h)), max(np.max(abs(l*w)),np.max(abs(h*(w-u)))),abs(gap)])
    maxima=np.maximum(maxima,err)
    assert np.max(err)<1e-8, err
for pulses in (s['reference_tracer_pulses'], None):
    a=dict(s,reference_tracer_pulses=pulses); out=m.solve_batch(a)
    cal=m._calibration(a['sensor_calibration_sweeps']); S,U=m._transport(a,cal); y=cal(a['y_obs_raw']); Z=a['Z_nuisance']
    P=np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+a['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    verify(P@S,P@y,a['source_ridge'],a['u_box'],a['total_budget'],out['kkt_certificate'])
    np.testing.assert_allclose(out['weights'],out['kkt_certificate']['weights']); np.testing.assert_allclose(out['terminal_field'],out['weights']@U.reshape(len(U),-1).copy() if False else np.einsum('i,ixy->xy',out['weights'],U))
    print('mode', 'tracer' if pulses else 'blind', 'weights',np.round(out['weights'],7),'field range',out['terminal_field'].min(),out['terminal_field'].max())
rng=np.random.default_rng(17)
for _ in range(320):
    n=int(rng.integers(2,30)); u=rng.choice([0.,.2,.4,1.],n); S=rng.normal(size=(int(rng.integers(2,70)),n)); y=rng.normal(size=len(S)); r=10.**rng.uniform(-5,1)
    for b in (0.,u.sum(),u[:n//2].sum(),u.sum()*rng.uniform()):
        verify(S,y,r,u,b,m._qp(S,y,r,u,b))
print('worst feasibility / stationarity / complementarity / dual gap',maxima)
np.set_printoptions(precision=6,suppress=True)
print('aux telemetry (public only):\n',s['aux_manifold_telemetry']); print('raw rosette first rows:\n',s['y_obs_raw'].reshape(-1,4)[:4]); print('single-bank sweeps', {k:(np.asarray(v).shape,np.asarray(v).ravel()[:5]) for k,v in s['sensor_calibration_sweeps']['single_bank_sweeps'].items()})
