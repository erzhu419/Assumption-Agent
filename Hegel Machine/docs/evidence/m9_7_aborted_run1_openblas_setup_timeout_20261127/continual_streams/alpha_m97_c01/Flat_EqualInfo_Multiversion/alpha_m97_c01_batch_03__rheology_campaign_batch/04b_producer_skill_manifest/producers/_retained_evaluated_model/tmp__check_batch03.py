import importlib.util, json, pickle
import numpy as np
p='/holdout_task/alpha_m97_c01_batch_03_solver.py'
spec=importlib.util.spec_from_file_location('solver',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_03.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; o=m.solve_alpha_m97_c01_batch_03(d)
rms=lambda x: np.sqrt(np.mean(x*x))
n=len(d['train']['lab_id']); t=d['eval_times']; dt=d['past_dt']; state=m._prepare(d); p,h,x=state
a,b=m._design(o['estimated_taus'],state,dt,t)
z=np.zeros((len(h),2)); tau=o['estimated_taus']
for j,width in enumerate(dt):
    q=np.exp(-width/tau); z=q*z+(1-q)*p[:,j,None]
wa=o['hereditary_weights']; wr=o['regime_mode_weights']; gap=h[:,None]-z; drift=x[:,0:5]@wa[2:]+wr[0]*np.sign(h)*np.sqrt(abs(h)); cross=wr[1]*(gap[:,0]-gap[:,1]); f=1-np.exp(-t[:,None]/tau)
assert np.allclose(o['predicted_strain'],d['test']['strain_0'][:,None]+((gap*wa[:2])@f.T+drift[:,None]*(t/(1+.85*t))+cross[:,None]*f[:,0])[n:])
assert np.allclose(o['initial_rate_pred'],((gap*wa[:2])@(1/tau)+drift+cross/tau[0])[n:])
for key,ref in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]:
    e=o[key]-g[ref]; s=d['test']; miss=~np.isfinite(s['coupon_features_raw']).all(1)
    print(key, 'complete/missing',[(int(k.sum()),rms(e[k])) for k in (~miss,miss)], 'by stratum',[rms(e[s['stratum']==j]) for j in np.unique(s['stratum'])], 'by hold sign',[rms(e[k]) for k in (h[n:]<0,h[n:]>=0)])
    print('worst lab/stratum',max((rms(e[k]),c,j,int(k.sum())) for c in range(4) for j in range(3) if (k:=(s['lab_id']==c)&(s['stratum']==j)).any()))
s=d['train']; delta=s['strain_curves_obs']-s['strain_0'][:,None]; y=s['instant_rate_obs']; print('leave-one-lab-out, observed response RMSE:')
for c in range(4):
    keep=s['lab_id']!=c; tr=tuple(v[:n][keep] for v in state); va=tuple(v[:n][~keep] for v in state)
    tau,w=m._fit(tr,dt,t,delta[keep],y[keep]); a,b=m._design(tau,va,dt,t)
    print(c,tau,rms(a@w-delta[~keep]),rms(b@w-y[~keep]))
cal=d['calibration_load_cells']; c0=np.flatnonzero(cal['lab_id']==0)
for name,keep,spikes in [('sparse', np.arange(len(cal['lab_id']))!=c0[-1], []),('EMI', np.ones(len(cal['lab_id']),bool), c0[:1])]:
    if name=='sparse': keep[c0[1:]]=False
    rows={k:v.copy() for k,v in cal.items()}; rows['true_stress'][spikes]+=5
    print(name,m._calibrate({k:v[keep] for k,v in rows.items()})[0])
