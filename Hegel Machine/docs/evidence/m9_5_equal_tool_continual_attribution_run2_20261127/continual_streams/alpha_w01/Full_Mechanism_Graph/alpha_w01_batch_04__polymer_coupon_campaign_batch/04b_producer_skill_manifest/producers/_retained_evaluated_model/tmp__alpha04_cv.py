import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w01_batch_04_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_alpha_w01_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; dt=d['past_dt']; time=d['eval_times']; sy=d['yield_stress']; maps=s._calibrate(d['calibration_load_cells']); N=len(lab)
err={p:np.empty_like(tr['strain_curves_obs']) for p in ('linear','gated','gated+imputation correction')}; rates={p:np.empty(N) for p in ('linear','gated')}; active=np.zeros(N,bool); missing=~np.isfinite(tr['coupon_features_raw']).all(axis=1)
def subset(b,keep): return {k:np.asarray(v)[keep] for k,v in b.items()}
def rmse(x): return np.sqrt(np.mean(x*x))
for k in range(4):
    fit=subset(tr,lab!=k); val=subset(tr,lab==k)
    panel={key:np.concatenate((val[key],d['test'][key])) for key in d['test'] if key in val}
    x, ok, n=s._alignment(fit,panel); keep=ok[:n]; fit=subset(fit,keep); xi=s._inputs(fit,x[:n][keep],maps,dt,sy); vi=s._inputs(val,x[n:n+len(val['lab_id'])],maps,dt,sy)
    loc=lab==k; active[loc]=(vi[-2]!=0)|(vi[-1]!=0)
    for p in ('linear','gated'):
        tau,w=s._fit(fit,xi,dt,time,plastic=(p=='gated')); a,b=s._design(tau,vi,dt,time)
        e=val['strain_0'][:,None]+a@w-val['strain_curves_obs']; r=b@w-val['instant_rate_obs']; err[p][loc]=e; rates[p][loc]=r
        if p=='gated':
            err['gated+imputation correction'][loc]=e-(r*missing[loc])[:,None]*time
            print('lab',k,'taus',np.round(tau,4),'plastic',np.round(w[7:],5),'curve',round(rmse(e),6),'rate',round(rmse(r),6),'gated count',active[loc].sum())
for p,e in err.items():
    slices={(int(k),int(m)):rmse(e[(lab==k)&(active==m)]) for k in range(4) for m in (False,True) if np.any((lab==k)&(active==m))}
    print(p,'pooled',rmse(e),'worst lab/yield',max(slices.items(),key=lambda item:item[1]),'missing',rmse(e[missing]),'worst horizon',max(rmse(e[:,i]) for i in range(len(time))))
for p,r in rates.items(): print(p,'worst lab rate',max(rmse(r[lab==k]) for k in range(4)))
e=err['gated+imputation correction']; r=rates['gated']
for label,m in [('complete',~missing),('missing',missing),('elastic',~active),('plastic',active)]:
    print(label,'curve abs q50/90/99/max',np.quantile(abs(e[m]),[.5,.9,.99,1]),'rate abs q50/90/99/max',np.quantile(abs(r[m]),[.5,.9,.99,1]),'curve mean by horizon',np.round(e[m].mean(0),4))
print('largest curve residuals (lab, horizon, residual):',[(int(lab[i]),float(time[j]),round(float(e[i,j]),3)) for i,j in np.dstack(np.unravel_index(np.argsort(abs(e).ravel())[-12:],e.shape))[0]])
print('complete residual cross-horizon correlation median',np.median(np.corrcoef(e[~missing],rowvar=False)[np.triu_indices(len(time),1)]))
print('archive/current maps')
import json
for sweep in json.loads(s._ARCHIVE.read_text()):
    print(sweep['episode_short_name'],np.round(np.array([s._supported_line(*s._checks(sweep,k)) for k in range(4)]),5).tolist())
print('current',np.round(maps,5).tolist())
