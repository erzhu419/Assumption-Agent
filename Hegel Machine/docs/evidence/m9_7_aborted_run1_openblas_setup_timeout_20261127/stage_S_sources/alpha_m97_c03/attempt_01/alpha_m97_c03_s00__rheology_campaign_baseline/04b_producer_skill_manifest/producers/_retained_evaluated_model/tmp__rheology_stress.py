import importlib.util, pickle, sys
import numpy as np
spec = importlib.util.spec_from_file_location('candidate',sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c03_s00.pkl','rb') as f:
    d=pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
cal = d['calibration_load_cells']
g=np.array([.81,1.17,.93,1.09]); b=np.array([-.21,.18,.09,-.04]);
cal['true_stress']=cal['cmd_stress']*g[cal['lab_id']]+b[cal['lab_id']]
n=len(tr['lab_id']); times=d['eval_times']
lab, groups, raw, panel = (np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons'))
x=raw.copy()
for j in range(4):
    i=lab==j; c=panel[i]; x[i]=(raw[i]-np.median(np.nanmean(c,axis=0)))/np.median(np.nanstd(c,axis=0))
    for s in np.unique(groups[i]):
        rows=i & (groups==s); z=x[rows]; x[rows]=np.where(np.isnan(z),np.nanmean(z,axis=0),z)
w = np.array([.74,.61,.18,-.15,.11,.19,-.12])
def analytic(p, a, taus):
    labs=p['lab_id']; z=np.zeros((len(labs),2)); h=p['stress_hold_cmd']*g[labs]+b[labs]
    for dt, cmd in zip(d['past_dt'],p['past_stress_cmd'].T):
        e=np.exp(-dt/taus); stress=cmd*g[labs]+b[labs]; z=e*z+(1-e)*stress[:,None]
    gap=h[:,None]-z; drift=w[2]+a@w[3:]
    return p['strain_0'][:,None] + (gap*w[:2])@(1-np.exp(-times[None,:]/taus[:,None])) + drift[:,None]*times, (gap*w[:2]/taus).sum(axis=1)+drift
rng=np.random.default_rng(92)
for taus in (np.array([.25,1.4]),np.array([1.1,6.2])):
    yc,yr=analytic(tr,x[:n],taus)
    mask=rng.random(yc.shape)<.07
    tr['strain_curves_obs']=yc+rng.normal(0,.015,yc.shape)+mask*rng.standard_cauchy(yc.shape)*2
    tr['instant_rate_obs']=yr+rng.normal(0,.018,yr.shape)
    o=m.solve_alpha_m97_c03_s00(d)
    yc,yr=analytic(te,x[n:],taus)
    ec=np.abs(o['predicted_strain']-yc); er=np.abs(o['initial_rate_pred']-yr)
    ts=np.max(np.abs(o['estimated_taus']/taus-1)); ws=np.max(np.abs(o['hereditary_weights']-w));
    worst_c=max(np.quantile(ec[te['lab_id']==j],.9) for j in range(4)); worst_r=max(np.quantile(er[te['lab_id']==j],.9) for j in range(4))
    print('injected taus',taus,'fit',o['estimated_taus'],'max relative tau error',ts,'max absolute weight error',ws,'worst lab p90 curve/rate error',worst_c,worst_r)
    assert ts<.12 and ws<.07 and worst_c<.035 and worst_r<.035 and np.all(o['regime_mode_weights']==0)
