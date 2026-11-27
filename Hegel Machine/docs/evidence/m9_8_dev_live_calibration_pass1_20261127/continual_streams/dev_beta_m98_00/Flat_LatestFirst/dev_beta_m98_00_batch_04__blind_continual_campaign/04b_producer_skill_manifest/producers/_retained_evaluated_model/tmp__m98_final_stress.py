import pickle, runpy, numpy as np
from scipy.optimize import minimize

m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
source = s['source_modes']; z = s['Z_nuisance']; cap = s['u_box']; budget = s['total_budget']; ridge = s['source_ridge']
cal = m['_calibration'](s['sensor_calibration_sweeps']); y = cal(s['y_obs_raw'])
S0, U0 = m['_response'](s, np.ones(8), source); original = m['solve_batch'](s); w0 = original['weights']
blind = dict(s, reference_tracer_pulses=None)
nom = m['solve_batch'](blind)
assert np.max(abs(nom['terminal_field']-original['terminal_field'])) < 1e-12, 'false blind wall detection on nominal public control'
print('nominal public with pulses masked: identical terminal field', flush=True)
# Identity raw coordinates preserve the measured residual on injected-wall cases.
x = s['sensor_calibration_sweeps']['ref_rosette']; blind['sensor_calibration_sweeps'] = dict(ref_rosette=x, raw_rosette=x)
pz = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
def project(a): return a-z@(pz@a)
def reference(S,U,y):
    A,b = project(S),project(y)
    def obj(w):
        r=A@w-b
        return .5*(r@r+ridge*(w@w))
    def grad(w): return A.T@(A@w-b)+ridge*w
    opt=minimize(obj, np.full(len(source),budget/len(source)), jac=grad, method='SLSQP', bounds=[(0,cap)]*len(source), constraints=[{'type':'eq','fun':lambda w:sum(w)-budget,'jac':lambda w:np.ones_like(w)}], options={'ftol':1e-14})
    assert opt.success, opt.message
    return opt.x@U.reshape(len(source),-1)
def err(a,b): return np.linalg.norm(a.ravel()-b)/np.linalg.norm(b)
for tag,wall in [('robin low',[.45,.45,1,1,.45,.45,1,1]),('robin high',[2,2,1,1,2,2,1,1]),('diffusion low',[1,1,.4,.4,1,1,.4,.4]),('diffusion high',[1,1,2.2,2.2,1,1,2.2,2.2]),('mixed',[.65,.65,1.6,1.6,.65,.65,1.6,1.6]),('anisotropic',[.55,.55,1,1,1.6,1.6,1,1]),('side',[.4,1.8,1,1,.75,1.4,1,1])]:
    S,U=m['_response'](s,np.array(wall),source)
    obs=y+(S-S0)@w0
    target=reference(S,U,obs)
    base=err(reference(S0,U0,obs),target)
    st=dict(blind,y_obs_raw=obs)
    out=m['solve_batch'](st)
    new=err(out['terminal_field'],target)
    cert=out['kkt_certificate']; w=out['weights']; gap=cert['primal_objective']-cert['dual_lower_bound']
    assert abs(w.sum()-budget)<1e-9 and w.min()>=0 and w.max()<=cap and abs(gap)<1e-9
    print(f'{tag}: field relative error {base:.5f} -> {new:.5f}; KKT gap {gap:.2e}',flush=True)
    if tag in ('robin low','diffusion low','mixed','side'):
        st['reference_tracer_pulses']=[dict(w_pulse=p['w_pulse'],y_pulse_raw=S@p['w_pulse']) for p in s['reference_tracer_pulses']]
        diagnosed=m['solve_batch'](st)
        pulse_error=err(diagnosed['terminal_field'],target)
        print(f'{tag} with diagnostic pulses: field relative error {pulse_error:.5e}',flush=True)
        assert pulse_error<1e-3
