import importlib.util, pickle, numpy as np
from scipy.optimize import minimize
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=m._Transport(s); decode=m._calibration(s)
p=s['reference_tracer_pulses']; b=m._pulse_losses(t,p,decode)
W=np.stack([i['w_pulse'] for i in p],axis=1); y=np.stack([decode(i['y_pulse_raw']) for i in p],axis=1)
print('tracer RMSE by shift and pulse, nominal then fitted')
for loss in (t.nominal,b):
    r=t.response(loss,t.sources@W)[0]-y
    print(np.sqrt(np.mean(r.reshape(len(t.shifts),len(t.sensors),len(p))**2,axis=1)))
for pulsed in (True,False):
    study=dict(s,reference_tracer_pulses=p if pulsed else None)
    out=m.solve_batch(study); w=out['weights']; cert=out['kkt_certificate'];lo=cert['lam_lower']; hi=cert['lam_upper']; u=s['u_box']
    S,U,_=t.response(b if pulsed else t.nominal,t.sources)
    Y=decode(s['y_obs_raw']); Z=s['Z_nuisance']; R=np.eye(len(Y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
    X=R@S; Y=R@Y; ridge=s['source_ridge']; grad=X.T@(X@w-Y)+ridge*w
    kkt=np.max(np.abs(grad+cert['nu_sum']-lo+hi)); gap=cert['primal_objective']-cert['dual_lower_bound']
    field_error=np.max(abs(out['terminal_field']-(U@w).reshape(t.nx,t.ny,order='F')))
    assert np.all(lo>=0) and np.all(hi>=0) and np.all(w>=0) and np.all(w<=u)
    assert np.array_equal(cert['weights'],w) and abs(w.sum()-s['total_budget'])<1e-12
    assert kkt<1e-11 and field_error<1e-12 and 0<=gap<1e-12 and max(abs(lo@w),abs(hi@(u-w)))<1e-12
    assert abs(cert['primal_objective']-0.5*(np.linalg.norm(X@w-Y)**2+ridge*(w@w)))<1e-12
    print('pulsed',pulsed,'KKT',kkt,'duality gap',gap,'terminal field max error',field_error)
rng=np.random.default_rng(1482); worst=np.zeros(4); counts=np.zeros(3,dtype=int)
for i in range(1800):
    n=int(rng.integers(2,34)); a=rng.normal(size=(n,n)); H=a.T@a/n+10**rng.uniform(-5,0)*np.eye(n); c=rng.normal(size=n)*10**rng.uniform(-1,2); cap=rng.choice([0.,.1,.3,.7,1.],n); budget=cap.sum()*rng.choice([0.,1.,1e-9,1-1e-9,rng.uniform()]) if i%3 else cap[rng.random(n)<.5].sum()
    w,nu=m._box_simplex(H,c,cap,budget); q=H@w-c+nu
    L=(w<=1e-13)&(cap>0); U=(w>=cap-1e-13)&(cap>0); F=~(L|U)&(cap>0)
    counts += [L.any(),U.any(),F.any()]; errors=[abs(w.sum()-budget),max(0.,np.max(-q[L],initial=0)),max(0.,np.max(q[U],initial=0)),np.max(abs(q[F]),initial=0)]
    worst=np.maximum(worst,errors)
    assert min(w)>=-1e-13 and max(w-cap)<=1e-13
assert max(worst)<1e-8
print('QP stress worst: budget, lower, upper, free',worst,'slice counts',counts)
cal=s['sensor_calibration_sweeps']; print('sweep fields',list(cal)); print('single bank fields and dimensions',[(k,np.shape(v),np.asarray(v).ravel()[:8].tolist()) for k,v in cal['single_bank_sweeps'].items()]); print('study fields',list(s)); print('banks and counts',np.unique(s['row_bank_id'],return_counts=True))
