import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
ds = item['dataset']; g = item['campaign_diagnostics']; train, test = ds['train'], ds['test']
t, dt = ds['eval_times'], ds['past_dt']
cal = ds['calibration_load_cells']; bridge = s._axial_load_map(cal)
loads, raw = cal['true_channel_loads'], cal['bridge_raw']
cal_errors = []
for fold in range(4):
    val = np.arange(len(loads)) % 4 == fold
    offset, axial = s._axial_load_map({'true_channel_loads': loads[~val], 'bridge_raw': raw[~val]})
    cal_errors.append(np.sqrt(np.mean(((raw[val]-offset)@axial-loads[val].sum(1))**2)))
print('Worst held-out calibration axial RMSE:', max(cal_errors))

out = s.solve_batch(ds); tau, theta = out['estimated_taus'], out['hereditary_weights']
assert .05 < tau[0] < tau[1] < 25
x = s._features(g,g)[0]; specimen = s._specimen(g,x,bridge)
c, r = s._branches(specimen,dt,t,tau); drift = specimen[2]@theta[2:]
y = g['strain_curves_obs']-g['strain_0'][:,None]; v = g['instant_rate_obs']
base_curve, base_rate = s._chamber_predict(specimen,dt,t,tau,theta)
step = 1e-5
p0, _ = s._chamber_predict(specimen,dt,np.array([0.]),tau,theta)
p1, _ = s._chamber_predict(specimen,dt,np.array([step]),tau,theta)
assert np.max(abs(p0)) < 1e-10 and np.max(abs((p1-p0).ravel()/step-base_rate)) < 5e-5
kernel = -np.expm1(-t)
# Isolate time-law error using the measured initial slope, not as a deployed input.
res = y-c@theta[:2]; anchored = v-r@theta[:2]
# Unit-slope bounded rational time law, compared out of lab against fixed exponential.
rat = np.empty_like(res); delta = np.empty_like(v)
for lab in np.unique(g['lab_id']):
    val = g['lab_id']==lab; fit=~val
    def rational(p):
        return t*(1+p[0]*t)/(1+p[1]*t+p[2]*t*t)
    p = least_squares(lambda p: (res[fit]-anchored[fit,None]*rational(p)).ravel(), [.5,1,.5],bounds=(0,np.inf),loss='soft_l1',f_scale=.004).x
    rat[val] = res[val]-anchored[val,None]*rational(p)
    # An added pairwise/history term must improve rate on a lab not used to fit it.
    pair = np.column_stack([x[:,i]-x[:,j] for i in range(x.shape[1]) for j in range(i+1,x.shape[1])] + [x[:,i]+x[:,j] for i in range(x.shape[1]) for j in range(i+1,x.shape[1])] + [r[:,0]-r[:,1]])
    mu=pair[fit].mean(0); scale=np.maximum(pair[fit].std(0),1e-10); a=(pair-mu)/scale
    b=np.linalg.solve(a[fit].T@a[fit]+np.eye(a.shape[1])*len(v[fit]), a[fit].T@(v-base_rate)[fit]); delta[val]=a[val]@b

def metric(e):
    return tuple(round(float(z),7) for z in (np.sqrt(np.mean(e**2)), max(np.sqrt(np.mean(e[g['lab_id']==k]**2)) for k in np.unique(g['lab_id']))))
print('Diagnostic pooled/worst lab RMSE:')
print('time law, rate anchored exponential / rational:', metric(res-anchored[:,None]*kernel), metric(rat))
print('deployed strain fixed / pairwise challenger:', metric(y-base_curve), metric(y-base_curve-delta[:,None]*kernel))
print('deployed rate fixed / pairwise challenger:', metric(v-base_rate), metric(v-base_rate-delta))
for lab in np.unique(g['lab_id']):
    val=g['lab_id']==lab
    print('lab',lab,'n',val.sum(),'mean signed strain/rate errors',np.mean((base_curve-y)[val]),np.mean((base_rate-v)[val]))
print('Train leave-one-lab-out, taus and worst strain/rate median absolute errors:')
xtr,_=s._features(train,test); bench=s._specimen(train,xtr,bridge)
bench_y=train['strain_curves_obs']-train['strain_0'][:,None]; bench_v=train['instant_rate_obs']; errors=[]
for lab in np.unique(train['lab_id']):
    val=train['lab_id']==lab
    tau_i,theta_i=s._fit(tuple(z[~val] for z in bench),dt,t,bench_y[~val],bench_v[~val]); hold=tuple(z[val] for z in bench)
    bc,br=s._branches(hold,dt,t,tau_i); d=hold[2]@theta_i[2:]
    errors.append((np.median(abs(bc@theta_i[:2]+d[:,None]*t-bench_y[val])),np.median(abs(br@theta_i[:2]+d-bench_v[val]))))
    print(lab,tau_i,errors[-1])
print('Worst train held-out lab strain/rate median absolute error:',np.max(errors,axis=0))
