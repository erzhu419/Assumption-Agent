import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch, _PAY
from _verified_upstream import calibrate_upstream
from downstream_solver import _basis, _chamber_features, _features, _loads, _predict_chamber
with open('/holdout_task/public_batch.pkl','rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
cal = calibrate_upstream(raw, _PAY)
di = calibrate_upstream({**raw, 'test':item['campaign_diagnostics']}, _PAY)['test']
out = solve_batch(raw)
taus, w = out['estimated_taus'], out['hereditary_weights']
t, dt = cal['eval_times'], cal['past_dt']
x = _chamber_features(di)
_, xjoint = _features(cal['train'], di)
past, hold = _loads(di, np.r_[0.,np.ones(4)])
b, r = _basis(taus, past, hold, x, dt, t)
base = di['strain_0'][:,None] + b[:,:,:2]@w[:2]
br = r[:,:2]@w[:2]
d = w[2] + x@w[3:]
y, q = di['strain_curves_obs'], di['instant_rate_obs']
lab = di['lab_id']; missing = np.isnan(di['coupon_features_raw']).any(1)
K = -np.expm1(-t)
actual, actual_r = _predict_chamber(di, taus,w,dt,t)
np.testing.assert_allclose(actual,base+d[:,None]*K,atol=1e-12)
np.testing.assert_allclose(actual_r,br+d,atol=1e-12)
# Fit any challenger parameters exclusively on other diagnostic labs. Local
# reference normalization and imputation use the label-free deployment panel.
pred = {k:[np.empty_like(y),np.empty_like(q)] for k in ('linear_joint','exp_joint','unit_exp','fitted_exp','rational_time','stress_pair','rational_drift')}
gap = hold-past[:,-1]
for j in np.unique(lab):
    tr = lab != j; te = ~tr
    dj = w[2]+xjoint@w[3:]
    def put(k, amp, kernel):
        pred[k][0][te] = base[te] + amp[te,None]*kernel
        pred[k][1][te] = br[te] + amp[te]
    put('linear_joint', dj, t)
    put('exp_joint', dj, K)
    put('unit_exp', d, K)
    for k, make in [('fitted_exp', lambda a: -a*np.expm1(-t/a)), ('rational_time',lambda a:t/(1+a*t*t))]:
        a = least_squares(lambda z: (base[tr]+d[tr,None]*make(z[0])-y[tr]).ravel(),[1.],bounds=(.001,20.),loss='soft_l1',f_scale=.003).x[0]
        put(k, d, make(a))
    # Test an omitted pairwise stress contribution to the drift amplitude,
    # and a nonlinear stress-dependent attenuation of the drift itself.
    for k, make, bounds in [('stress_pair', lambda a:d+a*gap,(-1.,1.)),('rational_drift',lambda a:d/(1+a*gap*gap),(0.,20.))]:
        def res(z):
            a=make(z[0])[tr]
            return np.r_[(base[tr]+a[:,None]*K-y[tr]).ravel(),br[tr]+a-q[tr]]
        a=least_squares(res,[0.],bounds=bounds,loss='soft_l1',f_scale=.003).x[0]
        put(k,make(a),K)
rmse=lambda a: float(np.sqrt(np.mean(a*a)))
for k,(c,rp) in pred.items():
    e, er = c-y,rp-q
    groups = [('lab'+str(j),lab==j) for j in np.unique(lab)]+[('complete',~missing),('missing',missing)]
    print(k, 'pooled',round(rmse(e),6),round(rmse(er),6),'worst coupon',round(max(map(rmse,e)),6),round(np.max(abs(er)),6),'slices',[(n,round(rmse(e[m]),6),round(rmse(er[m]),6)) for n,m in groups])
