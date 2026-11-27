import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s0 = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
rng = np.random.default_rng(32)
x = s0['sensor_calibration_sweeps']['ref_rosette']
a = np.eye(4) + rng.normal(0, .08, (4, 4))
b = np.array([.023, -.036, .04, .01])
def transform(z, mode):
    x = np.asarray(z).reshape(-1,4)
    if mode == 'identity':
        return np.asarray(z).copy()
    y = x @ a + b
    for j in range(4):
        p,q,r = (j+1)%4,(j+2)%4,(j+3)%4
        if mode == 'cubic':
            y[:,j] += .19*x[:,p]*x[:,q] + .12*x[:,p]*x[:,q]*x[:,r]
        if mode == 'rational':
            y[:,j] += .19*x[:,p]*x[:,q]/(1+.8*(x[:,r]-.4*x[:,j])**2)
    return y.reshape(np.shape(z))
source = s0['source_modes']
w0 = np.array([.42,.24,0,.08,.42,0,.24,0])
W = np.array([p['w_pulse'] for p in s0['reference_tracer_pulses']])
Z = s0['Z_nuisance']; rho = s0['nuisance_ridge']; R = np.eye(len(Z))-Z @ np.linalg.solve(Z.T@Z+rho*np.eye(Z.shape[1]),Z.T)
max_gap = 0
for mode, wall, pulses in [('identity',[1]*8,False),('affine',[1]*8,True),('cubic',[.75,.75,1,1,1.55,1.55,1,1],True),('affine',[.7,1.4,.52,1.25,1.6,.65,1.3,.6],True),('rational',[1]*8,True)]:
    s = dict(s0)
    s['sensor_calibration_sweeps'] = dict(s0['sensor_calibration_sweeps'],raw_rosette=transform(x,mode)+rng.normal(0,1e-5,x.shape))
    S, U = m._response(s, np.asarray(wall), source)
    yc = S @ w0 + Z @ [.02,-.015,.06] + rng.normal(0,.005,len(Z))
    s['y_obs_raw'] = transform(yc,mode)
    s['reference_tracer_pulses'] = [dict(w_pulse=w,y_pulse_raw=transform(S @ w,mode)) for w in W] if pulses else None
    out = m.solve_batch(s)
    S2,y2 = R @ S,R @ yc
    ridge = s['source_ridge']
    H, f = S2.T @ S2 + ridge*np.eye(len(w0)), S2.T @ y2
    opt = minimize(lambda w: .5*w@H@w-f@w+w[0]*0.,np.ones(len(w0))*s['total_budget']/len(w0),jac=lambda w: H@w-f,method='SLSQP',bounds=[(0,s['u_box'])]*len(w0),constraints=[dict(type='eq',fun=lambda w:w.sum()-s['total_budget'],jac=lambda w:np.ones_like(w))],options={'ftol':1e-14,'maxiter':300})
    cert=out['kkt_certificate']; gap=cert['primal_objective']-cert['dual_lower_bound']; max_gap=max(max_gap,gap)
    cal=m._calibration(s['sensor_calibration_sweeps']); fitted_wall = m._walls(s,cal,source)
    g=S2.T@(S2@out['weights']-y2)+ridge*out['weights']
    stat=g-cert['nu_sum']-np.array(cert['lam_lower'])+np.array(cert['lam_upper'])
    print(mode,'pulses',pulses,'cal max',np.max(abs(cal(s['y_obs_raw'])-yc)),'wall max',np.max(abs(fitted_wall-wall)),'weight max',np.max(abs(out['weights']-opt.x)),'field max',np.max(abs(out['terminal_field']-np.tensordot(opt.x,U,axes=1))),'oracle KKT residual',np.max(abs(stat)),'internal gap',gap)
    assert opt.success and abs(sum(out['weights'])-s['total_budget']) < 1e-9 and abs(gap)<1e-8
print('max internal gap',max_gap)
for cap,total in [(.3,0.),(.3,.3),(.3,1.7),(.3,2.4)]:
    X=np.eye(8); y=np.arange(8.)
    w=m._weights(X,y,.04,cap,total)
    g=1.04*w-y; nu=m._dual(g,w,cap); d=g-nu
    gap=np.maximum(d,0)@w+np.maximum(-d,0)@(cap-w)
    print('active-face total',total,'gap',gap,'budget error',abs(w.sum()-total))
    assert abs(w.sum()-total)<1e-9 and gap<1e-8
archive=Path('/tmp/hegel_continual_archive/archive_apparatus_log.json')
print('apparatus log present:',archive.is_file())
if archive.is_file():
    with archive.open() as f: print(f.read(3500))
