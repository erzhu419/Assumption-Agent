import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_m97_c01_batch_02_solver.py'
s=importlib.util.spec_from_file_location('cand',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_02.pkl','rb'))[0]['dataset']
t=d['train']; dt=d['past_dt']; time=d['eval_times']; n=len(t['lab_id']); state=tuple(a[:n] for a in m._prepare(d)); y=t['strain_curves_obs']-t['strain_0'][:,None]; r=t['instant_rate_obs']; lab=t['lab_id']; missing=~np.isfinite(t['coupon_features_raw']).all(axis=1)
q=lambda a: np.round(np.quantile(np.abs(a),[.5,.8,.9,.95,.99,1.]),5)
for loss in ('cauchy','soft_l1','linear'):
    e=np.zeros_like(y); f=np.zeros_like(r); ps=[]
    for j in range(4):
        ok=lab!=j; test=~ok
        par,w=m._fit(tuple(a[ok] for a in state),dt,time,y[ok],r[ok],loss=loss)
        A,B=m._design(par,tuple(a[test] for a in state),dt,time)
        e[test]=A@w-y[test]; f[test]=B@w-r[test]; ps.append(np.round(np.exp(par),4))
    print(loss,'taus',ps,'abs errors curve/rate',q(e),q(f),flush=True)
    print('lab curve median, worst-horizon median, rate median:',[(j,round(float(np.median(np.abs(e[lab==j]))),5),round(float(np.max(np.median(np.abs(e[lab==j]),axis=0))),5),round(float(np.median(np.abs(f[lab==j]))),5)) for j in range(4)],flush=True)
    print('complete/missing curve/rate:',[(q(e[ss])[:3],q(f[ss])[:3]) for ss in (~missing,missing)],flush=True)
    if loss=='cauchy':
        print('curve horizon scaled MAD',np.round([m._scale(v) for v in e.T],5),'signed median',np.round(np.median(e,axis=0),5),flush=True)
        print('rate scaled MAD',round(m._scale(f),5),'by stratum',[(int(j),q(f[t['stratum']==j])[:3]) for j in np.unique(t['stratum'])],flush=True)
        print('curve signed median by lab',[(j,np.round(np.median(e[lab==j],axis=0),4)) for j in range(4)],flush=True)
