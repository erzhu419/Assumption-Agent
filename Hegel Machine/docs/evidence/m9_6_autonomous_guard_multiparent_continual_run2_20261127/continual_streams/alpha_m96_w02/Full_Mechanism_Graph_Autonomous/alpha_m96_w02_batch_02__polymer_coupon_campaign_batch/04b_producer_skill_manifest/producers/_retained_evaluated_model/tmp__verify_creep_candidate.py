import importlib.util, pickle, time
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = '/holdout_task/'
m = load('solver', base + 'alpha_m96_w02_batch_02_solver.py')
import ast, math, types
source = base + 'validate_public_schema.py'
with open(source) as f:
    tree = ast.parse(f.read(), filename=source)
fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify_box_simplex_kkt_certificate')
v = types.ModuleType('validator_function')
v.np, v.math = np, math
exec(compile(ast.Module(body=[fn], type_ignores=[]), source, 'exec'), v.__dict__)
for e in pickle.load(open(base + 'public_alpha_m96_w02_batch_02.pkl','rb')):
    d = e.get('dataset', e.get('data', e)); tr,te = d['train'],d['test']; gt=e['gt']
    start=time.monotonic(); o=m.solve_alpha_m96_w02_batch_02(d); elapsed=time.monotonic()-start
    pred=o['predicted_strain']; r=pred-gt['te_clean']; lab=te['lab_id']; missing=(~np.isfinite(te['coupon_features_raw'])).any(axis=1)
    rms=lambda a: float(np.sqrt(np.mean(a*a)))
    print('seconds',round(elapsed,4),'tau abs error',abs(o['estimated_taus']-gt['taus']),'strain RMSE',rms(r),'worst horizon',max(map(rms,r.T)),'allocation max abs error',max(abs(o['allocation_weights']-gt['w_alloc_opt'])))
    for c in range(4):
        print('public lab',c,'all/complete/missing RMSE',[round(rms(r[j]),6) if j.any() else None for j in (lab==c,(lab==c)&~missing,(lab==c)&missing)])
    x=m._align(tr,te)[:len(tr['lab_id'])]; cal=m._stress_calibration(d['calibration_load_cells']); dt,t=d['past_dt'],d['eval_times']; inp=m._inputs(tr,x,cal)
    y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; lab=tr['lab_id']; missing=(~np.isfinite(tr['coupon_features_raw'])).any(axis=1)
    keys=('coupon_features_raw','instant_rate_obs')
    for c in range(4):
        j=lab==c; i=~j; design=m._Design(dt,t,tuple(a[i] for a in inp)); tau,b=m._fit(design,y[i],rate[i]); a=m._Design(dt,t,tuple(a[j] for a in inp)).make(tau); p=a[:,:-1]@b
        delta=m._rate_adjustment({k:tr[k][i] for k in keys},{k:tr[k][j] for k in keys},design,a,tau,b); q=missing[j]; assert np.all(delta[~q]==0)
        print('LOLO lab',c,'missing RMSE before/after',[round(rms(err[q]),6) for err in (p-y[j],p+delta[:,None]*t-y[j])])
    z=d['Z_nuisance']; pz=np.linalg.solve(z.T@z+d['nuisance_ridge']*np.eye(z.shape[1]),z.T); s=d['S_alloc_modes']*(1+.25*np.tanh(pred[:,-1]-te['strain_0']))[:,None]; s=s-z@(pz@s); yy=d['y_alloc_obs']; yy=yy-z@(pz@yy); u=d['u_box']
    gaps=[]
    for B in (0.,.35*u,u,d['total_budget'],6*u,11.8*u,12*u):
        w,cert=m._allocation(dict(d,total_budget=B),pred); checked=v.verify_box_simplex_kkt_certificate(s,yy,d['source_ridge'],u,B,cert)
        assert checked['valid'], (B,checked)
        g=s.T@(s@w-yy)+d['source_ridge']*w-cert['nu_sum']; lo=cert['lam_lower']; hi=cert['lam_upper']; assert max(abs(g-lo+hi))<1e-6 and max(abs(lo*w))<1e-7 and max(abs(hi*(u-w)))<1e-7, B
        gaps.append(checked['gap'])
    print('worst verified KKT gap across seven budgets',max(gaps))
