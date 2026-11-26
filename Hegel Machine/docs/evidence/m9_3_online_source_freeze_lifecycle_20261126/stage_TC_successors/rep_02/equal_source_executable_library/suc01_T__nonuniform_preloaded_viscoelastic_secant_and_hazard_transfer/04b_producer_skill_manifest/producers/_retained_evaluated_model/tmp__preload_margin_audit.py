import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def profile(a, t, dt):
    # Subtract the observed initial slope to eliminate each specimen's unknown drift.
    y = a['strain_curves_obs'] - a['strain_0'][:,None] - a['instant_rate_obs'][:,None]*t
    scale = np.hypot(.018, .012*t)
    def residual(p):
        fast, gap = p[:2]
        tau = np.array([fast, fast + (9-fast)*gap])
        amp = m._deficit(tau, dt, a['past_stress'], a['initial_memory_states'], a['stress_hold_0'])*p[2:]
        shape = -np.expm1(-t[:,None]/tau)-t[:,None]/tau
        return ((amp @ shape.T-y)/scale).ravel()
    fits = []
    for fast,slow in [(.45,3.),(.8,6.),(.27,1.5)]:
        bounds=([.2,1e-7,0,0],[8.999,1.,np.inf,np.inf])
        start=[fast,(slow-fast)/(9-fast),1.,1.]
        f=least_squares(residual,start,bounds=bounds,loss='soft_l1',f_scale=2.5)
        fits.append(least_squares(residual,f.x,bounds=bounds,loss='cauchy',f_scale=1.5))
    fast,gap,*gain=min(fits,key=lambda f:f.cost).x
    return np.array([fast,fast+(9-fast)*gap]), np.array(gain)

def amp(a, tau, gain, dt):
    return m._deficit(tau,dt,a['past_stress'],a['initial_memory_states'],a['stress_hold_0'])*gain

def predict(a,b,t,dt,method):
    x,y,keep,well=m._aligned_features(a,b)
    if method=='saved':
        tau,gain,coef=m._fit(a,t,dt,x,keep)
    else:
        tau,gain=profile(a,t,dt)
        r=a['instant_rate_obs']-amp(a,tau,gain,dt)@(1/tau)
        coef=least_squares(lambda c:x[keep]@c-r[keep],np.linalg.lstsq(x[keep],r[keep],rcond=None)[0],loss='soft_l1',f_scale=.02).x
    A=amp(b,tau,gain,dt)
    rate=b['instant_rate_obs']-A@(1/tau)
    flow=rate+.7*well*(y@coef-rate)
    pred=b['strain_0'][:,None]+A@(-np.expm1(-t[:,None]/tau)).T+flow[:,None]*t
    return tau,pred

for u in pickle.load(open('public_suc01_T_datasets.pkl','rb')):
    d,g=u['dataset'],u['gt']; a,b=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']
    w=g['theta']; tau=g['taus']; A=amp(b,tau,w[:2],dt); flow=w[2]+g['te_Z']@w[3:]
    closure=b['strain_0'][:,None]+A@(-np.expm1(-t[:,None]/tau)).T+flow[:,None]*t
    print('algebraic closure: max strain and derivative errors',np.max(abs(closure-g['te_clean'])),np.max(abs(A@(1/tau)+flow-g['te_true_rate0'])))
    for method in ('saved','profile'):
        tau,p=predict(a,b,t,dt,method)
        o=m.solve_preloaded_hereditary_transfer(d) if method=='saved' else None
        if o is not None: np.testing.assert_allclose(p,o['predicted_strain'])
        h=m._crossing_times(t,b['strain_0'],p,b['critical_strain'])
        sec=(p[:,-1]-b['strain_0'])/np.maximum(abs(b['stress_hold_0']),.5)
        err=np.c_[abs(p-g['te_clean']).mean(1),abs(sec-g['te_secant']),abs(h-g['te_cross'])]
        print(method,'tau rel',abs(tau/g['taus']-1),'public overall',err.mean(0))
        cell=b['lab_id']*3+b['stratum']; preload=np.linalg.norm(b['initial_memory_states'],axis=1); hold=b['stress_hold_0']; idx=np.arange(len(hold));
        for label,v in [('lab/stratum',cell),('preload top quartile',preload>=np.quantile(preload,.75)),('hold denominator floor',abs(hold)<.5),('missing coupon',~np.isfinite(b['coupon_features_raw']).all(1))]:
            print(label,[(int(j),int(sum(v==j)),err[v==j].mean(0).round(5).tolist()) for j in np.unique(v)])
        for lab in np.unique(a['lab_id']):
            k=a['lab_id']==lab; tr={key:v[~k] for key,v in a.items()}; te={key:v[k] for key,v in a.items()}
            tt,pp=predict(tr,te,t,dt,method); r=pp-te['strain_curves_obs']; clipped=np.minimum(abs(r),.2)
            print('LOLO',lab,'max tau rel',round(max(abs(tt/g['taus']-1)),5),'curve clipped MAE',round(clipped.mean(),5),'terminal median signed residual by stratum',[round(np.median(r[te['stratum']==j,-1]),5) for j in np.unique(te['stratum'])])
