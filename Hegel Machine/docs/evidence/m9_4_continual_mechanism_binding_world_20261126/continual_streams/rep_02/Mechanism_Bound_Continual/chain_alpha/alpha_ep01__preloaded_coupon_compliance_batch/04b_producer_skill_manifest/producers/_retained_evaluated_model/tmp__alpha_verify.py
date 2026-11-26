import importlib.util, pickle, numpy as np
from time import perf_counter
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
cases = pickle.load(open('/holdout_task/public_alpha_ep01.pkl', 'rb'))

def independent_crossing(start, path, crit, times):
    if start >= crit:
        return 0.
    for i, end in enumerate(path):
        if end >= crit:
            xp = [start if i == 0 else path[i-1], end]
            fp = [0. if i == 0 else times[i-1], times[i]]
            return np.interp(crit, xp, fp)
    return times[-1]

for k, case in enumerate(cases):
    d = case['dataset']; train, test = d['train'], d['test']; times = d['eval_times']; dt = d['past_dt']
    clock = perf_counter(); result = s.solve_alpha_ep01(d); elapsed = perf_counter() - clock
    tau = result['estimated_taus']; assert np.all(tau > 0) and np.all(np.diff(tau) > 0)
    p = result['predicted_strain']; c = d['calibration_load_cells']; lab = test['lab_id']
    fits = np.array([np.polyfit(c['cmd_stress'][c['lab_id']==j], c['true_stress'][c['lab_id']==j], 1) for j in range(4)])
    hold = fits[lab,0] * test['stress_hold_cmd'] + fits[lab,1]
    np.testing.assert_allclose(result['secant_compliance'], (p[:,-1]-test['strain_0']) / np.maximum(abs(hold), .5), atol=1e-12)
    cross = np.array([independent_crossing(a,b,c,times) for a,b,c in zip(test['strain_0'],p,test['critical_strain'])])
    np.testing.assert_allclose(result['hazard_crossing_times'], cross, atol=1e-12)
    x, complete = s._align(d); n=len(train['lab_id']); gain,bias = s._calibrate(c)
    rec = test['initial_memory_states'].copy()
    past = fits[lab,0,None] * test['past_stress_cmd'] + fits[lab,1,None]
    for stress, step in zip(past.T, dt):
        rec += -np.expm1(-step/tau) * (stress[:,None] - rec)
    mat = s._Design(test,x[n:],gain,bias,dt,times).matrix(np.log(tau))
    np.testing.assert_allclose(mat[:,:-1,:2], (hold[:,None]-rec)[:,None,:] * -np.expm1(-times[:,None]/tau), atol=1e-12)
    print('case',k,'tau',tau.round(6),'runtime_s',round(elapsed,3),'crossing_initial/interpolated/final',[(cross==0).sum(),((cross>0)&(cross<times[-1])).sum(),(cross==times[-1]).sum()])
    y=np.column_stack((train['strain_curves_obs']-train['strain_0'][:,None],train['instant_rate_obs']))
    for j in range(4):
        val=train['lab_id']==j; mask=complete[:n] & ~val
        log_t,coef=s._fit(s._Design(train,x[:n],gain,bias,dt,times,mask),y[mask]); panel={key: v[val] for key,v in train.items()}
        design=s._Design(panel,x[:n][val],gain,bias,dt,times)
        pred=s._predict(design,log_t,coef,panel,complete[:n][val],times)
        obs=panel['strain_curves_obs']; err=abs(pred-obs)
        subsets=[complete[:n][val],~complete[:n][val]]
        print('  lab',j,'fold_tau',np.exp(log_t).round(5),'capped curve MAE complete / missing',*[round(np.minimum(err[m],.10).mean(),5) if m.any() else None for m in subsets],'secant median AE',round(np.median(err[:,-1]/np.maximum(abs(design.hold),.5)),5))

start=np.array([1., 0., 0., 0., 0.]); times=np.array([1., 2., 4.]); paths=np.array([[0,0,0],[2,0,2],[0,1,0],[0,0,0],[0,0,1]])
np.testing.assert_allclose(s._crossings(start, paths, np.ones(5), times), [0.,.5,2.,4.,4.])
print('Independent recurrence, all readouts, and crossing edge cases passed.')
