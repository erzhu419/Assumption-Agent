import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; t=d['train']; dt=d['past_dt']; times=d['eval_times']
a,_=m._prepare(d); y=t['strain_curves_obs']-t['strain_0'][:,None]; r=t['instant_rate_obs']; ta,th=m._fit(dt,times,a,y,r)
def score(name, holdout):
    take=~holdout
    tau,th0=m._fit(dt,times,tuple(z[take] for z in a),y[take],r[take]); c,v=m._design(tau,dt,times,tuple(z[holdout] for z in a)); e=c@th0-y[holdout]; f=v@th0-r[holdout]
    print(f'{name}: n={holdout.sum()} taus={tau.round(6)} max_rel_tau_shift={np.max(abs(tau/ta-1)):.4g} max_weight_shift={np.max(abs(th0-th)):.4g} curve_medabs={np.median(abs(e)):.6g} curve_p90={np.quantile(abs(e),.9):.6g} curve_rmse={np.sqrt(np.mean(e**2)):.6g} rate_medabs={np.median(abs(f)):.6g} rate_rmse={np.sqrt(np.mean(f**2)):.6g}')
for lab in np.unique(t['lab_id']): score(f'lab={lab}',t['lab_id']==lab)
for i,ix in enumerate(np.array_split(np.argsort(a[1]),4)):
    h=np.zeros(len(r),bool); h[ix]=True; score(f'hold_stress_quartile={i}',h)
o=m.solve_batch(d); print('public material',ta,th); print('deployment range',[(k,float(o[k].min()),float(o[k].max())) for k in ('predicted_strain','initial_rate_pred')])
