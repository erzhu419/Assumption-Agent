import importlib.util, pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch07', root+'alpha_w03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_w03_batch_07.pkl', 'rb'))[0]['dataset']
out = m.solve_alpha_w03_batch_07(d)
shapes = {'estimated_taus': (2,), 'hereditary_weights': (7,), 'viscoplastic_weights': (2,), 'predicted_strain': (80,10), 'initial_rate_pred': (80,)}
assert set(out) == set(shapes)
assert all(out[k].shape == s and np.isfinite(out[k]).all() for k,s in shapes.items())
np.testing.assert_array_equal(out['viscoplastic_weights'], [0,0])
rng = np.random.default_rng(713)
dt, t = d['past_dt'], d['eval_times']
gain = np.array([1.13,.82,1.29,.91]); bias = np.array([.13,-.16,.105,-.06])
lab = np.repeat(np.arange(4),8); cmd = np.tile(np.linspace(-2,2,8),4)
cal = dict(lab_id=lab,cmd_stress=cmd,true_stress=gain[lab]*cmd+bias[lab])
tau = np.array([.58,4.1]); coef = np.array([.72,.49,.08,.15,-.13,.11,-.09,.24,-.037]); sy = 1.35
# Alternate signs within each lab, independently of the interleaved lab IDs.
def coupons(n):
    l = np.arange(n)%4
    offset, scale = (l-1.5)*3, .7+.4*l
    x = rng.normal(size=(n,4)); h = rng.uniform(-2.8,2.8,n); p = rng.uniform(-2.4,2.4,(n,len(dt)))
    v = np.zeros((n,2)); work = np.zeros(n)
    for j,delta in enumerate(dt):
        relax = np.exp(-delta/tau)
        v = relax*v + (1-relax)*p[:,j,None]
        work += delta*np.maximum(abs(p[:,j])-sy,0)**2
    amp = (h[:,None]-v)*coef[:2]
    amp[:,0] += coef[8]*work*h
    drift = coef[2]+x@coef[3:7]+coef[7]*np.maximum(abs(h)-sy,0)**2*np.sign(h)
    rate = np.sum(amp/tau,axis=1)+drift
    strain0 = rng.normal(size=n)
    curves = strain0[:,None]+amp@(1-np.exp(-t[None,:]/tau[:,None]))+drift[:,None]*t
    panel = offset[:,None]+scale[:,None]*(2*(np.arange(n)//4%2)-1)[:,None]*np.ones((1,32))
    b = dict(lab_id=l,stratum=np.arange(n)//4%3,coupon_features_raw=offset[:,None]+scale[:,None]*x,ref_coupons=panel,past_stress_cmd=(p-bias[l,None])/gain[l,None],strain_0=strain0,stress_hold_cmd=(h-bias[l])/gain[l],instant_rate_obs=rate)
    return b,curves,rate,h,work,x,p
tr,tr_curve,tr_rate,tr_h,tr_work,tr_x,tr_p = coupons(192); te,curve,rate,h,work,x,p = coupons(64)
tr['strain_curves_obs'] = tr_curve
synth = dict(train=tr,test=te,past_dt=dt,eval_times=t,yield_stress=sy,calibration_load_cells=cal)
aligned, phys_h, phys_p = m._prep(synth)
np.testing.assert_allclose(aligned[:, 1:],np.vstack((tr_x,x)),atol=1e-12)
np.testing.assert_allclose(phys_h,np.r_[tr_h,h],atol=1e-12)
np.testing.assert_allclose(phys_p,np.vstack((tr_p,p)),atol=1e-12)
c,r = m._design(tau,aligned[-len(x):],h,p,dt,t,sy)
np.testing.assert_allclose(te['strain_0'][:,None]+c@coef,curve,atol=1e-12)
np.testing.assert_allclose(r@coef,rate,atol=1e-12)
te['coupon_features_raw'][::9,2] = np.nan
s = m.solve_alpha_w03_batch_07(synth)
errs=[]
for key, expected in [('estimated_taus',tau),('hereditary_weights',coef[:7]),('viscoplastic_weights',coef[7:]),('predicted_strain',curve),('initial_rate_pred',rate)]:
    err = np.max(abs(s[key]-expected)); errs.append(err); print(key,'active-case max error',f'{err:.3g}')
for name, mask in [('positive yield',h>sy),('negative yield',h<-sy),('history only',(abs(h)<=sy)&(work>0)),('missing feature',np.isnan(te['coupon_features_raw']).any(axis=1))]:
    assert mask.any()
    print(name,mask.sum(),'max curve error',np.max(abs(s['predicted_strain'][mask]-curve[mask])))
assert max(errs) < 1e-6
